from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import os
import pandas as pd

client = OpenAI(
      api_key=""
)

df = pd.read_csv('venki_new.csv')

current_date_time = datetime.now()
formatted_date_time = current_date_time.strftime("%m-%d-%H-%M")

exp_name = 'experiments/gpt-4o-final/samples_175/baseline/' + formatted_date_time
os.makedirs(exp_name, exist_ok=True)


for i in tqdm(range(175)):

   prompt = "Legal Context:\n" + df['Context'][i] + "\n" \
"Question:\n" + df['Question'][i] + "\n\n" \
+ df['Options'][i] \

   prompt = prompt + "\n\nProvide the response in the following format: \nExplanation: [Legal reasoning step by step as numbered points]\nFinal answer: [Final answer as an English capital letter from the options given above]"

   conversation = [
         {"role": "system", "content": "You are an expert legal assistant."},
         {"role": "user", "content": prompt}
      ]

   # print(prompt)

   final_output = client.chat.completions.create(
      model="gpt-4o",
      messages=conversation,
      temperature=0.0,
      seed=42,
   ).choices[0].message.content

   # print(final_output)

   # break

   with open(exp_name + '/response_' + str(i+1) + '.txt', 'w') as f:
      # f.write(completion.choices[0].message.content)
      # I want to write generated_reasoning_steps, generated_reasoning_errors, final_output

      f.write(final_output)

   # break