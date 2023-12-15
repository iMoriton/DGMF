from typing import List, Optional
import fire
from llama import Llama, Dialog
import re
import os
import time
from dall_e3 import *

def merge_specific_txt_files(folder_path: str,
                             output_file: str,
                             file_prefix: Optional[str] = None,
                             file_count: Optional[int] = None,
                             file_list: Optional[List[str]] = None):
    output_file = os.path.join(folder_path, output_file)
    if file_list is None:
        file_list = [f"{file_prefix}{i}.txt" for i in range(file_count)]
    with open(output_file, 'w') as outfile:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
            except FileNotFoundError:
                print(f"File not found: {file_path}")


def extract_bracket_contents(text):
    # Regular expression pattern to match all characters between { and }
    # Using non-greedy matching (.*?) to ensure each pair of brackets is matched separately
    pattern = r'\{(.*?)\}'

    # Use the findall method to find all matches
    matches = re.findall(pattern, text)

    return matches


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        word_count: int = 5000,
        seed: int = 1,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
        word_count (int, optimal): The number of words in the output content.
        seed (int, optimal): Seed used by the model.
    """
    folder_path = f"./results/results_word_{word_count}_seed_{seed}"
    os.makedirs(folder_path, exist_ok=True)
    word_count //= 2
    labels = ['Introduction', 'Related work', 'Methodology', 'Results', 'Future work', 'Conclusion']
    percentage = (1 / len(labels)) ** 2
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed,
    )
    dialogs_0: List[Dialog] = [
        [
            {
                "role": "system",
                "content": "Always break down the paper writing task into these subtasks: introduction, related work, methodology, results, future work and conclusion. Always describe each subtask in one sentence and use curly braces '{}' around each description. For instance: {Introduction: A concise overview of the paper's main points, highlighting the significance of the study, the research questions, and the main findings}.",
            },
            {"role": "user",
             "content": "Please help me write an academic paper, titled \"Political Mobilization in the Digital Era: The Impact of Online Activities on Traditional Politics\".",
             },
        ],
    ]
    start_time = time.time()
    results_0 = generator.chat_completion(
        dialogs_0,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    cost = time.time() - start_time
    with open(f"{folder_path}/time.csv", 'a') as outfile:
        outfile.write(f"main_task, {cost:.2f}\n")
    for dialog_0, result_0 in zip(dialogs_0, results_0):
        for msg in dialog_0:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result_0['generation']['role'].capitalize()}: {result_0['generation']['content']}"
        )
        print("\n==================================\n")
        subtasks_1 = extract_bracket_contents(result_0['generation']['content'])
        all_subtasks_1 = ""
        for i in range(len(subtasks_1)):
            all_subtasks_1 = all_subtasks_1 + f"{i + 1}. {subtasks_1[i]}\n"
        print(all_subtasks_1)
        for i in range(len(subtasks_1)):
            dialogs_1: List[Dialog] = [
                [
                    {
                        "role": "system",
                        "content": f"You are a professional paper writer. Next, you will get a main task of writing a paper and a subtask of it. You need to generate 6 sub-subtasks of this subtask, always describe each sub-task in one sentence and output like this {{{labels[i]} 1: ...}}, {{{labels[i]} 2: ...}}, {{{labels[i]} 3: ...}}, {{{labels[i]} 4: ...}}, {{{labels[i]} 5: ...}}, {{{labels[i]} 6: ...}}, be sure to output the curly braces, i.e. \"{{}}\"",
                    },
                    {
                        "role": "user",
                        "content": f"""The main task is \"{dialogs_0[0][1]['content']}\" and the subtask is {subtasks_1[i]}."""
                    },
                ],
            ]
            start_time = time.time()
            results_1 = generator.chat_completion(
                dialogs_1,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            cost = time.time() - start_time
            with open(f"{folder_path}/time.csv", 'a') as outfile:
                outfile.write(f"task{i}, {cost:.2f}\n")
            print('\n\n')
            for dialog_1, result_1 in zip(dialogs_1, results_1):
                print(f"{dialog_1[0]['role'].capitalize()}: {dialog_1[0]['content']}")
                print(f"> {result_1['generation']['role'].capitalize()}: {result_1['generation']['content']}")
                subtasks_2 = extract_bracket_contents(result_1['generation']['content'])
                all_subtasks_2 = ""
                for j in range(len(subtasks_2)):
                    all_subtasks_2 = all_subtasks_2 + f"{j + 1}. {subtasks_2[j]}\n"
                print(all_subtasks_2)
                for j in range(len(subtasks_2)):
                    dialogs_2: List[Dialog] = [
                        [
                            {
                                "role": "system",
                                "content": f"You are a professional paper writer. Next, you will get a main task of writing a paper and a subtask of it and a sub-subtask of the subtask. You only need to complete the writing of the sub-subtask. Your output for this sub-subtask must be at least {int(percentage * word_count)} words. ",
                            },
                            {
                                "role": "user",
                                "content": f"""The main task is \"{dialogs_0[0][1]['content']}\" and the subtask is {subtasks_1[i]} and the sub-subtask is {subtasks_2[j]}. Please output in the following format:
                                {labels[i]} {j + 1}:...(output at least {int(percentage * word_count)} words.)
                                """
                            },
                        ],
                    ]
                    start_time = time.time()
                    results_2 = generator.chat_completion(
                        dialogs_2,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    cost = time.time() - start_time
                    with open(f"{folder_path}/time.csv", 'a') as outfile:
                        outfile.write(f"task{i}_{j}, {cost:.2f}\n")
                    print(results_2[0]['generation']['content'])
                    with open(f"{folder_path}/result{i}_{j}.txt", "w") as file:
                        file.write(results_2[0]['generation']['content'])
                    print('\n\n')
                merge_specific_txt_files(folder_path=folder_path, output_file=f"result{i}.txt",
                                         file_prefix=f"result{i}_", file_count=len(subtasks_2))
            if i == 2 or i == 3 :
                with open(os.path.join(folder_path, f'result{i}.txt'), 'r', encoding='utf-8') as file:
                    text = file.read()
                dialogs: List[Dialog] = [
                    [
                        {
                            "role": "system",
                            "content": "You're a skilled essay assistant, adept at condensing essay content. The user will supply a section of their essay, and your task is to summarize it into a succinct summary of absolutely no more than 100 words. Always encase this summary within curly brackets i.e. \'{}\' and output it in the following format:\n{Summary: ......(a succinct summary of 50 to 150 words)}",
                        },
                        {"role": "user",
                         "content": f"Please compress the following text:\n\"\"\"\n{text}\n\"\"\", and encase this summary within curly brackets i.e. \'{{}}\' and output it.",
                         },
                    ],
                ]
                start_time = time.time()
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                cost = time.time() - start_time
                with open(os.path.join(folder_path, f'summary_time.csv'), 'a', encoding='utf-8') as file:
                    file.write(f'summary{i},{cost}\n')
                print(f'Generation time: {cost}')
                print(results[0]['generation']['content'])
                content = extract_bracket_contents(results[0]['generation']['content'])
                content = content[0]
                encoded_content = content.encode('utf-8')
                byte_count = len(encoded_content)
                with open(os.path.join(folder_path, f'{labels[i]}.txt'), 'w', encoding='utf-8') as file:
                    file.write(content)
                print(content, byte_count)
                generate_picture(folder_path, f'{labels[i]}', i)
        merge_specific_txt_files(folder_path=folder_path, output_file="result.txt",
                                 file_list=[f"result{i}.txt" for i in range(len(labels))])


if __name__ == "__main__":
    fire.Fire(main)
