#!/usr/bin/env python3
"""Local model judge for generated Repliqa summaries."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SummaryJudge:
    """Evaluate a generated summary with a local chat model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def evaluate(
        self,
        *,
        document_id: str,
        summary: str,
        questions: list[dict[str, Any]],
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        """Return a structured quality judgment for one generated summary.

        The judge is called twice. The first call cannot see reference answers and
        only produces generated answers from the summary. The second call scores
        those generated answers against the reference answers.
        """
        answer_result = self.answer_questions(
            document_id=document_id,
            summary=summary,
            questions=questions,
            max_new_tokens=max_new_tokens,
        )
        score_result = self.score_answers(
            document_id=document_id,
            generated_answers=answer_result["generated_answers"],
            questions=questions,
            max_new_tokens=max_new_tokens,
        )
        normalized = self._normalize_result(score_result["parsed_output"])
        normalized["answer_raw_output"] = answer_result["raw_output"]
        normalized["score_raw_output"] = score_result["raw_output"]
        normalized["generated_answers"] = answer_result["generated_answers"]
        normalized["judge_model"] = str(self.model_path)
        return normalized

    def answer_questions(
        self,
        *,
        document_id: str,
        summary: str,
        questions: list[dict[str, Any]],
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        """Answer questions using only the generated summary."""
        messages = self.answer_messages(
            document_id=document_id,
            summary=summary,
            questions=questions,
        )
        raw_output = self._chat(messages, max_new_tokens=max_new_tokens)
        parsed = self._parse_json(raw_output)
        generated_answers = self._normalize_generated_answers(parsed)
        return {
            "raw_output": raw_output,
            "parsed_output": parsed,
            "generated_answers": generated_answers,
        }

    def score_answers(
        self,
        *,
        document_id: str,
        generated_answers: list[dict[str, str]],
        questions: list[dict[str, Any]],
        max_new_tokens: int = 512,
    ) -> dict[str, Any]:
        """Score generated answers against reference answers."""
        messages = self.score_messages(
            document_id=document_id,
            generated_answers=generated_answers,
            questions=questions,
        )
        raw_output = self._chat(messages, max_new_tokens=max_new_tokens)
        parsed = self._parse_json(raw_output)
        return {
            "raw_output": raw_output,
            "parsed_output": parsed,
        }

    @staticmethod
    def answer_messages(
        *,
        document_id: str,
        summary: str,
        questions: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided generated summary. "
                    "You cannot see the original document or the reference answers. "
                    "If the summary does not contain enough information to answer a "
                    "question, output exactly N/A for that question. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": SummaryJudge.answer_prompt(
                    document_id=document_id,
                    summary=summary,
                    questions=questions,
                ),
            },
        ]

    @staticmethod
    def answer_prompt(
        *,
        document_id: str,
        summary: str,
        questions: list[dict[str, Any]],
    ) -> str:
        question_payload = [
            {
                "question_id": str(question.get("question_id", "")),
                "question": str(question.get("question", "")),
            }
            for question in questions
        ]
        return f"""Answer each question for document {document_id} using only the generated summary.

Rules:
- Do not use outside knowledge.
- Do not infer from the reference answer; it is not provided in this step.
- If the generated summary does not contain enough information to answer a question, set generated_answer to exactly "N/A".
- Otherwise, answer in one concise sentence.

Return only this JSON object:
{{
  "generated_answers": [
    {{
      "question_id": "...",
      "question": "...",
      "generated_answer": "answer from the summary, or N/A"
    }}
  ]
}}

Generated summary:
{summary}

Questions:
{json.dumps(question_payload, ensure_ascii=False, indent=2)}
"""

    @staticmethod
    def score_messages(
        *,
        document_id: str,
        generated_answers: list[dict[str, str]],
        questions: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict answer evaluator. Compare generated answers "
                    "against reference answers. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": SummaryJudge.score_prompt(
                    document_id=document_id,
                    generated_answers=generated_answers,
                    questions=questions,
                ),
            },
        ]

    @staticmethod
    def score_prompt(
        *,
        document_id: str,
        generated_answers: list[dict[str, str]],
        questions: list[dict[str, Any]],
    ) -> str:
        reference_payload = [
            {
                "question_id": str(question.get("question_id", "")),
                "question": str(question.get("question", "")),
                "reference_answer": str(question.get("answer", "")),
            }
            for question in questions
        ]
        return f"""Score generated answers for document {document_id}.

For each question, compare generated_answer with reference_answer.

Rules:
- If the answers mostly match, assign score 1.0 and covered=true.
- If they do not match, assign score 0.0 and covered=false.
- If reference_answer is "N/A", "NA", "Not available", "Not mentioned", or equivalent, the correct generated_answer is exactly "N/A".
- If reference_answer is N/A and generated_answer is exactly "N/A", assign score 1.0 and covered=true.
- If reference_answer is N/A but generated_answer is concrete, assign score 0.0, covered=false, and add the generated answer to unsupported_claims.
- If generated_answer is N/A but reference_answer contains a concrete answer, assign score 0.0 and covered=false.

Return only this JSON object:
{{
  "answer_coverage": 0.0,
  "factual_consistency": 0.0,
  "completeness": 0.0,
  "conciseness": 0.0,
  "final_score": 0.0,
  "question_scores": [
    {{
      "question_id": "...",
      "score": 0.0,
      "covered": false,
      "generated_answer": "generated answer",
      "reference_answer": "reference answer",
      "reason": "brief reason"
    }}
  ],
  "missing_evidence": ["brief missing item"],
  "unsupported_claims": ["brief unsupported claim"]
}}

Generated answers:
{json.dumps(generated_answers, ensure_ascii=False, indent=2)}

Reference questions and answers:
{json.dumps(reference_payload, ensure_ascii=False, indent=2)}
"""

    def _chat(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _parse_json(raw_output: str) -> dict[str, Any]:
        parsed = SummaryJudge._decode_json_object(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError("judge output must be a JSON object")
        return parsed

    @staticmethod
    def _decode_json_object(raw_output: str) -> Any:
        text = raw_output.strip()
        if not text:
            raise ValueError("judge output is empty")

        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        candidates.extend(chunk.strip() for chunk in fenced if chunk.strip())

        decoder = json.JSONDecoder()
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

            for match in re.finditer(r"\{", candidate):
                try:
                    parsed, _ = decoder.raw_decode(candidate[match.start() :])
                    return parsed
                except json.JSONDecodeError:
                    continue

        match = re.search(r"(\{[\s\S]*\})", text)
        if match:
            return json.loads(match.group(1))
        raise json.JSONDecodeError("No JSON object found", raw_output, 0)

    @staticmethod
    def _normalize_generated_answers(parsed: dict[str, Any]) -> list[dict[str, str]]:
        raw_answers = parsed.get("generated_answers", [])
        if not isinstance(raw_answers, list):
            raise ValueError("answer-step output must include generated_answers list")
        generated_answers: list[dict[str, str]] = []
        for item in raw_answers:
            if not isinstance(item, dict):
                continue
            answer = str(item.get("generated_answer", "")).strip()
            generated_answers.append(
                {
                    "question_id": str(item.get("question_id", "")),
                    "question": str(item.get("question", "")),
                    "generated_answer": answer or "N/A",
                }
            )
        if not generated_answers:
            raise ValueError("answer-step output contains no generated answers")
        return generated_answers

    @staticmethod
    def _normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
        def score(name: str) -> float:
            value = float(parsed.get(name, 0.0))
            return max(0.0, min(1.0, value))

        question_scores = parsed.get("question_scores", [])
        if isinstance(question_scores, list) and question_scores:
            valid_scores: list[float] = []
            for item in question_scores:
                if not isinstance(item, dict):
                    continue
                try:
                    valid_scores.append(max(0.0, min(1.0, float(item.get("score", 0.0)))))
                except (TypeError, ValueError):
                    valid_scores.append(1.0 if item.get("covered") is True else 0.0)
            answer_coverage = (
                sum(valid_scores) / len(valid_scores)
                if valid_scores
                else score("answer_coverage")
            )
        else:
            answer_coverage = score("answer_coverage")

        return {
            "answer_coverage": max(0.0, min(1.0, answer_coverage)),
            "factual_consistency": score("factual_consistency"),
            "completeness": score("completeness"),
            "conciseness": score("conciseness"),
            "final_score": score("final_score"),
            "question_scores": question_scores,
            "missing_evidence": parsed.get("missing_evidence", []),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }


def load_summary_and_questions(
    result_json: str | Path,
) -> tuple[str, str, list[dict[str, Any]], Path | None, dict[str, Any]]:
    result_path = Path(result_json)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    state_path = Path(result["message_center_state"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    document_id = str(state["document"]["document_id"])
    summary = str(result.get("final_summary") or "")
    questions = state.get("questions", [])
    trajectory_path = Path(result["trajectory_path"]) if result.get("trajectory_path") else None
    if not summary:
        raise ValueError(f"final_summary is empty in {result_path}")
    if not questions:
        raise ValueError(f"questions are empty in {state_path}")
    workflow = {
        "submitted": bool(result.get("submitted", False)),
        "summary_rounds_used": int(result.get("summary_rounds_used", 0)),
        "max_summary_rounds": int(result.get("max_summary_rounds", 1)),
        "summary_chars": len(summary),
    }
    return document_id, summary, questions, trajectory_path, workflow


def covered_question_count(question_scores: list[Any], total_questions: int) -> int:
    """Count reference questions covered by the generated summary."""
    covered = 0
    for item in question_scores:
        if not isinstance(item, dict):
            continue
        if item.get("covered") is True:
            covered += 1
            continue
        try:
            if float(item.get("score", 0.0)) >= 0.5:
                covered += 1
        except (TypeError, ValueError):
            pass
    return min(covered, total_questions)


def submit_score(answer_score: float, submitted: bool) -> float:
    """Reward submitting only when answer coverage is strong enough."""
    if submitted:
        if answer_score >= 0.8:
            return 1.0
        if answer_score >= 0.6:
            return 0.5
        return 0.0
    if answer_score < 0.6:
        return 0.7
    if answer_score < 0.8:
        return 0.4
    return 0.0


def round_score(used_rounds: int, max_rounds: int) -> float:
    """Prefer fewer rounds, without letting efficiency dominate quality."""
    if max_rounds <= 1:
        return 1.0
    used = max(1, min(used_rounds, max_rounds))
    return max(0.0, min(1.0, 1.0 - ((used - 1) / (max_rounds - 1))))


def length_score(char_count: int) -> float:
    """Soft reward for summaries that are neither too short nor too long."""
    if char_count < 200:
        return 0.2
    if char_count < 400:
        return char_count / 400.0
    if char_count <= 1200:
        return 1.0
    if char_count <= 2000:
        return 1.0 - ((char_count - 1200) / 800.0)
    return 0.0


def compute_training_reward(
    *,
    judge_result: dict[str, Any],
    total_questions: int,
    submitted: bool,
    used_rounds: int,
    max_rounds: int,
    summary_chars: int,
) -> dict[str, Any]:
    """Compute the training reward from fixed, inspectable components."""
    answered = covered_question_count(
        judge_result.get("question_scores", []),
        total_questions=total_questions,
    )
    answer = answered / max(total_questions, 1)
    submit = submit_score(answer, submitted)
    rounds = round_score(used_rounds, max_rounds)
    length = length_score(summary_chars)
    reward = (0.65 * answer) + (0.15 * submit) + (0.10 * rounds) + (0.10 * length)
    reward = max(0.0, min(1.0, reward))
    return {
        "final_score": round(reward, 6),
        "components": {
            "answer_score": round(answer, 6),
            "answered_questions": answered,
            "total_questions": total_questions,
            "submit_score": round(submit, 6),
            "submitted": submitted,
            "round_score": round(rounds, 6),
            "summary_rounds_used": used_rounds,
            "max_summary_rounds": max_rounds,
            "length_score": round(length, 6),
            "summary_chars": summary_chars,
        },
        "weights": {
            "answer_score": 0.65,
            "submit_score": 0.15,
            "round_score": 0.10,
            "length_score": 0.10,
        },
    }


def update_trajectory_reward(trajectory_path: Path, judge_result: dict[str, Any]) -> None:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    final_score = float(judge_result["final_score"])
    trajectory["reward"] = final_score
    metrics = trajectory.setdefault("metrics", {})
    metrics["judge_answer_coverage"] = judge_result["answer_coverage"]
    metrics["judge_factual_consistency"] = judge_result["factual_consistency"]
    metrics["judge_completeness"] = judge_result["completeness"]
    metrics["judge_conciseness"] = judge_result["conciseness"]
    metrics["judge_final_score"] = final_score
    components = judge_result.get("reward_components", {})
    if isinstance(components, dict):
        for name, value in components.items():
            if isinstance(value, int | float | bool):
                metrics[f"reward_{name}"] = value
    trajectory_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one generated summary locally.")
    parser.add_argument("result_json", type=Path, help="Path to *.multi_round_result.json.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/yaojiaxin/RL/models/Qwen3-14B"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--update-trajectory-reward",
        action="store_true",
        help="Write final_score back to the saved trajectory reward field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    document_id, summary, questions, trajectory_path, workflow = load_summary_and_questions(
        args.result_json
    )
    judge = SummaryJudge(args.model_path)
    result = judge.evaluate(
        document_id=document_id,
        summary=summary,
        questions=questions,
        max_new_tokens=args.max_new_tokens,
    )
    judge_model_final_score = result["final_score"]
    reward = compute_training_reward(
        judge_result=result,
        total_questions=len(questions),
        submitted=workflow["submitted"],
        used_rounds=workflow["summary_rounds_used"],
        max_rounds=workflow["max_summary_rounds"],
        summary_chars=workflow["summary_chars"],
    )
    result["judge_model_final_score"] = judge_model_final_score
    result["final_score"] = reward["final_score"]
    result["reward_components"] = reward["components"]
    result["reward_weights"] = reward["weights"]
    result["document_id"] = document_id
    result["result_json"] = str(args.result_json)
    result["trajectory_path"] = str(trajectory_path) if trajectory_path else None

    if args.update_trajectory_reward:
        if trajectory_path is None:
            raise ValueError("result file does not include trajectory_path")
        update_trajectory_reward(trajectory_path, result)
        result["trajectory_reward_updated"] = True
    else:
        result["trajectory_reward_updated"] = False

    output = args.output
    if output is None:
        output = args.result_json.with_suffix("")
        output = output.with_name(output.name + ".judge_result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"judge_result_path": str(output), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
