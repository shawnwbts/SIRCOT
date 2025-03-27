from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import random
import time
import math

random.seed(1234)

client = OpenAI(
    api_key='your_api_key',
    base_url = "your_url"
)

def generate_summaries_scence(code_snippet, summary):
    sysContent = 'Analyze the code and summarization.' + code_snippet + summary
    content = 'Just  output one word from seven words: Functional,Input/Output,Security,Boundary,Logic,Error Handling, External Interfaces.'
    try:
        response = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': sysContent},
                {'role': 'user', 'content': content},
            ],
            model='gpt-3.5-turbo',
            temperature=0.2,
            stream=True
        )
        generated_text = ""
        seen_words = set()
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                if content.strip() not in seen_words:
                    seen_words.add(content.strip())
                    generated_text += content
        return generated_text
    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_dynamic_cot(code_snippet, summary):
    task_templates = {
        "Functional":
            "Briefly Identify the function's purpose(reference only).",
        "Input/Output":
            "Briefly Identify the inputs and outputs(reference only).",
        "Security":
            "Briefly Identify security checks or validation(reference only).",
        "Boundary":
            "Briefly Identify the limitations of inputs and outputs(reference only).",
        "Logic":
            "Briefly Identify core algorithm or logic(reference only).",
        "Error Handling":
            "Briefly Identify error handling(reference only).",
        "External Interfaces":
            "Briefly Identify external APIs or libraries the function interacts with(reference only)."
    }

    task_type = generate_summaries_scence(code_snippet, summary)
    cot = task_templates.get(task_type)
    if cot is None:
        print(f"Error: No matching task type found for {task_type}")
        cot = "None"
    return cot

def generate_summaries_chain_of_thought(example_source1, example_target1, example_source2, example_target2,
                                         example_source3, example_target3, example_source4, example_target4, code,
                                         example_choose):
    sysContent1 = f"Generate a short summarization in one sentence for smart contract code. To alleviate the difficulty of this task, I will give you four examples to learn."
    prompt1 = '#example code 1:' + example_source1 + \
             '\n#example summarization 1:' + example_target1 + \
             '\n#example code 2:' + example_source2 + \
             '\n#example summarization 2:' + example_target2 + \
             '\n#example code 3:' + example_source3 + \
             '\n#example summarization 3:' + example_target3 + \
             '\n#example code 4:' + example_source4 + \
             '\n#example summarization 4:' + example_target4

    cot_comment = generate_dynamic_cot(code, example_choose)

    sysContent2 = f"Now, analyze the following contract.(very Main Instruction)"
    prompt2 = code + "Instruction reference briefly (very Secondary):" + cot_comment
    prompt3 = f"Organize the above information to the point and generate a short summarization in one sentence. "+'Generated summarization(The length must not exceed [' + example_choose + ']):\n'

    # print(prompt1 + prompt2 + prompt3)

    try:
        response = client.chat.completions.create(
            messages=[{'role': 'system', 'content': sysContent1},
                      {'role': 'user', 'content': prompt1},
                      {'role': 'system', 'content': sysContent2},
                      {'role': 'user', 'content': prompt2},
                      {'role': 'user', 'content': prompt3}],
            model='gpt-3.5-turbo',
            temperature=0.2,
            stream=True
        )
        generated_text = ""

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                generated_text += content
        return generated_text

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    df = pd.read_csv('data/example_all.csv')
    example_code1 = df['code1'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment1 = df['comment1'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code2 = df['code2'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment2 = df['comment2'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code3 = df['code3'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment3 = df['comment3'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code4 = df['code4'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment4 = df['comment4'].tolist()

    df = pd.read_csv('data/test_data_function.csv', header=None,encoding='ISO-8859-1')
    source_codes = df[0].tolist()

    df = pd.read_csv('data/test_data_comment.csv', header=None,encoding='ISO-8859-1')
    example = df[0].tolist()

    batch_size = 50

    num_batches = math.ceil(len(source_codes) / batch_size)
    print(num_batches)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(source_codes))

        source_batch = source_codes[start_index:end_index]
        example_batch = example[start_index:end_index]

        python_codes = []
        for i in tqdm(range(len(source_batch)), mininterval=0.1, maxinterval=1):
            python_codes.append(generate_summaries_chain_of_thought(example_code1[i], example_comment1[i], example_code2[i],example_comment2[i],example_code3[i],example_comment3[i],example_code4[i],example_comment4[i],source_batch[i], example_batch[i]))
        time.sleep(1)
        # print(python_codes)
        df = pd.DataFrame(python_codes)
        if batch_index == 0:
            with open('result/sml1.csv', 'w', newline='\n') as f:
                df.to_csv(f, index=False, header=True)
        else:
            with open('result/sml1.csv', 'a', newline='\n') as f:
                df.to_csv(f, index=False, header=True)