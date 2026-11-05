"""One model, bounded correction, immutable requests; no credentials in receipts."""
import os
import re
import time
from pathlib import Path
import json
import jsonschema
import requests
from .io import frozen, read, write, value_hash


class Provider:
    def __init__(self, config):
        self.config = config
        self.repository = Path(config["repository"])

    def ask(self, phase, packet, folder, audit, recovery_id=None):
        folder = Path(folder)
        if recovery_id is None or not (folder / "FAILED.json").exists() or (folder / "RESULT.json").exists():
            return self._ask_once(phase, packet, folder, audit)
        if not re.fullmatch(r"attempt_[0-9]+", recovery_id):
            raise ValueError("A request recovery must cite its explicit campaign attempt")
        previous = sorted((folder / "recoveries").glob("*/REPAIR.json"))
        for repair_path in reversed(previous):
            if (repair_path.parent / "RESULT.json").exists():
                repair = read(repair_path)
                return self._ask_once(phase, packet, repair_path.parent, audit, repair["context"])
        destination = folder / "recoveries" / recovery_id
        repair_path = destination / "REPAIR.json"
        if repair_path.exists():
            context = read(repair_path)["context"]
        else:
            if len(previous) >= 2:
                raise ValueError("Request recovery limit reached; user intervention is required")
            parent = previous[-1].parent if previous else folder
            responses = sorted(parent.glob("RESPONSE_*.json"))
            submission = None
            if responses:
                try:
                    submission = json.loads(read(responses[-1]).get("content") or "")
                except ValueError:
                    pass
            reason = read(parent / "FAILED.json")["reason"] if (parent / "FAILED.json").exists() else "Unfinished request"
            if submission is not None:
                try:
                    audit(submission)
                except (ValueError, KeyError) as error:
                    reason = str(error)
            context = dict(previous_submission=submission, deterministic_audit=reason,
                instruction="Return a complete corrected JSON submission. Copy exact supplied identifiers; do not invent or shorten evidence IDs.")
            frozen(repair_path, dict(parent_request=str(parent), parent_request_sha256=value_hash(read(parent / "REQUEST.json")),
                                    campaign_attempt=recovery_id, recovery_number=len(previous) + 1, context=context))
        return self._ask_once(phase, packet, destination, audit, context)

    def _ask_once(self, phase, packet, folder, audit, repair_context=None):
        folder = Path(folder)
        skill_name = "parent-candidate-screening" if phase == "screen" else "interaction-fragment-combination"
        skill = self.repository / "skills" / skill_name
        texts = [(skill / "SKILL.md").read_text(encoding="utf-8"),
                 (skill / "references/contract.md").read_text(encoding="utf-8")]
        schema = read(self.repository / "schemas" / f"{phase}.schema.json")
        system = "\n\n".join(texts) + "\nReturn one JSON object matching this JSON Schema:\n" + json.dumps(schema)
        binding = dict(model=self.config["model"], api_base=self.config["api_base"],
                       packet=packet, system_sha256=value_hash(system), schema=schema,
                       max_tokens=self.config["llm_output_tokens"])
        if repair_context is not None:
            binding["repair_context"] = repair_context
        frozen(folder / "REQUEST.json", binding)
        messages = [dict(role="system", content=system), dict(role="user", content=json.dumps(packet))]
        if repair_context is not None:
            messages.append(dict(role="user", content=json.dumps(dict(repair_context=repair_context))))
        last_error = None
        for correction in range(2):
            response_path = folder / f"RESPONSE_{correction:02d}.json"
            if response_path.exists():
                response = read(response_path)
            else:
                secret = os.environ.get("DEEPSEEK_API_KEY")
                if not secret:
                    raise RuntimeError("DEEPSEEK_API_KEY is unavailable in the process environment")
                payload = dict(model=self.config["model"], messages=messages,
                               response_format=dict(type="json_object"), max_tokens=self.config["llm_output_tokens"],
                               stream=False)
                for transport_attempt in range(3):
                    try:
                        transport = requests.post(self.config["api_base"].rstrip("/") + "/chat/completions",
                            headers={"Authorization": "Bearer " + secret}, json=payload,
                            timeout=(25, self.config["llm_timeout_seconds"]))
                        if transport.status_code in (429, 500, 502, 503, 504) and transport_attempt < 2:
                            time.sleep(2 ** transport_attempt)
                            continue
                        if transport.status_code != 200:
                            write(folder / "TRANSPORT_FAILED.json", dict(status=transport.status_code, attempt=transport_attempt))
                            raise RuntimeError(f"DeepSeek request failed with HTTP {transport.status_code}")
                        raw = transport.json()
                        choice = raw["choices"][0]
                        response = dict(model=raw.get("model"), request_id=raw.get("id"), usage=raw.get("usage"),
                                        finish_reason=choice.get("finish_reason"), content=choice["message"].get("content"))
                        break
                    except (requests.Timeout, requests.ConnectionError) as error:
                        if transport_attempt == 2:
                            write(folder / "TRANSPORT_FAILED.json", dict(error_type=type(error).__name__, attempt=transport_attempt))
                            raise RuntimeError("DeepSeek transport failed after three bounded attempts") from None
                write(response_path, response)
            try:
                if response.get("finish_reason") != "stop":
                    raise ValueError("Incomplete model response")
                result = json.loads(response.get("content") or "")
                jsonschema.validate(result, schema)
                audit(result)
                frozen(folder / "RESULT.json", result)
                return result
            except (ValueError, KeyError, jsonschema.ValidationError) as error:
                last_error = str(error)[:3500]
                write(folder / f"AUDIT_{correction:02d}.json", dict(valid=False, reason=last_error))
                messages += [dict(role="assistant", content=response.get("content") or "{}"),
                             dict(role="user", content="Correct your JSON submission using this audit feedback. Keep every source and protocol unchanged. " + last_error)]
        write(folder / "FAILED.json", dict(reason=last_error, correction_calls=1))
        raise ValueError("LLM submission remains invalid after one correction: " + last_error)


def audit_screen(result, candidates):
    expected = {candidate["identifier"]: candidate for candidate in candidates}
    identifiers = [decision["candidate_identifier"] for decision in result["decisions"]]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected):
        raise ValueError("Screen decisions must cover every supplied candidate exactly once")
    errors = []
    for decision in result["decisions"]:
        if decision["evidence_ids"] != [expected[decision["candidate_identifier"]]["evidence_id"]]:
            errors.append(dict(candidate_identifier=decision["candidate_identifier"],
                               required_evidence_ids=[expected[decision["candidate_identifier"]]["evidence_id"]],
                               received_evidence_ids=decision["evidence_ids"]))
    if errors:
        raise ValueError("Copy the exact source evidence for each mismatched candidate: " + json.dumps(errors))


def audit_fusions(result, parents, regions, config):
    from .regions import compile_condition
    parent_map = {parent["identifier"]: parent for parent in parents}
    region_map = {region["identifier"]: region for region in regions}
    pairs, identifiers = set(), set()
    for proposal in result["proposals"]:
        names = [entry["parent_identifier"] for entry in proposal["parent_regions"]]
        pair = tuple(sorted(names))
        if len(set(names)) != 2 or pair in pairs or proposal["identifier"] in identifiers:
            raise ValueError("Repeated parent pair, parent contribution, or proposal identifier")
        pairs.add(pair)
        identifiers.add(proposal["identifier"])
        condition = compile_condition(proposal, parents, regions, config)
        expected_evidence = {parent_map[name]["evidence_id"] for name in names}
        allowed_evidence = set(expected_evidence)
        for entry in proposal["parent_regions"]:
            selected_evidence = {evidence for identifier in entry["region_identifiers"]
                                 for evidence in region_map[identifier]["interaction_evidence"]}
            if not selected_evidence.intersection(proposal["evidence_ids"]):
                raise ValueError("Each parent must cite a selected interaction")
            allowed_evidence.update(selected_evidence)
        actual = set(proposal["evidence_ids"])
        if not expected_evidence <= actual or not actual <= allowed_evidence:
            raise ValueError("Unbound source or interaction evidence")
    remaining_pairs = {tuple(sorted([left["identifier"], right["identifier"]]))
                       for position, left in enumerate(parents) for right in parents[position + 1:]} - pairs
    excluded = [tuple(sorted(entry["parents"])) for entry in result["unselected_pairs"]]
    if len(excluded) != len(set(excluded)) or set(excluded) != remaining_pairs:
        raise ValueError("Account for each unselected unordered parent pair exactly once")
