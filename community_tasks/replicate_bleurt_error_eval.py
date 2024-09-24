"""
# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401

Example evaluation script to replicate BLEURT lambda error.

python -m lighteval accelerate \
    --model_args "dummy" \
    --tasks "community|summarization_bleurt_example|0|0" \
    --custom_tasks ./community_tasks/replicate_bleurt_error_eval.py \
    --max_samples 3 \
    --output_dir "./results/" \
    --override_batch_size 1
"""
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def prompt_fn(line, task_name: str = None):
    custom_query = f"\nLegal decision: {line['text']}\nSummarize the above legal decision in the language of {line['language']}.\n"
    return Doc(
        task_name=task_name,
        query=custom_query,
        choices=[str(line["regeste"])],
        gold_index=0,
        specific={
            "law_area": line["law_area"],
            "text": line["text"],
            "regeste": line["regeste"],
            "language": line["language"],
        },
    )


# reference: https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/lighteval_task.py#L65
task = LightevalTaskConfig(
    name="summarization_bleurt_example",
    suite=["community"],
    prompt_function=prompt_fn,
    hf_repo="rcds/swiss_leading_decision_summarization",
    hf_subset="",  # to specify subset, replace with "de", "it", or "en"
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["validation"],  # ["validation", "test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=10,
    metric=[
        Metrics.bleurt,
    ],
    stop_sequence=["\n"],
    trust_dataset=True,
)

TASKS_TABLE = [task]

if __name__ == "__main__":
    print(f"Task names: {[task.name for task in TASKS_TABLE]}")
    print(f"Number of tasks: {len(TASKS_TABLE)}")
