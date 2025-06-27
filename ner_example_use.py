import torch
from transformers import pipeline
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_DIR = os.path.join(script_dir, "models", "NER models", "final_model")

# Check if a GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Setting up inference pipeline...")
ner_pipeline = pipeline(
    "ner", model=FINAL_MODEL_DIR, tokenizer=FINAL_MODEL_DIR,
    device=0 if device == 'cuda' else -1
)

adverse_media_texts = [
    # "The SEC has charged former CEO of Enron, Jeffrey Skilling, with fraud.",
    # "A fine was issued to Deutsche Bank by the UK authorities for failing to prevent money laundering.",
    # "Mr. John Smith, a resident of Germany, was implicated in a bribery case involving a government contract.",
    '''The founder of the world's largest crypto exchange has been sentenced to four months in prison in the US for allowing criminals to launder money on his platform.
Changpeng Zhao resigned from Binance in November and pleaded guilty to violating US money laundering laws.

Binance was ordered to pay $4.3bn (£3.4bn) after a US investigation found it helped users bypass sanctions.

Prosecutors had sought a three-year sentence for the former Binance boss.

At a sentencing hearing on Tuesday in Seattle, Judge Richard Jones said Zhao put "Binance's growth and profits over compliance with US laws and regulations", according to the Verge.

US officials said in November that Binance and Zhao's "wilful violations" of its laws had threatened the US financial system and national security.

"Binance turned a blind eye to its legal obligations in the pursuit of profit," said Treasury Secretary Janet Yellen.

"Its wilful failures allowed money to flow to terrorists, cybercriminals, and child abusers through its platform."

What is Bitcoin? Key cryptocurrency terms explained
Richard Teng: Who is Binance's new boss?
Commonly called "CZ", Zhao has a $33 billion fortune, according to Forbes magazine.

Nigerian authorities are currently investigating the company, registered in the Cayman Islands, as well.

Tigran Gambarayan, who is in charge of financial crime compliance at Binance, denied money laundering charges in a Nigerian court in early April.

Fellow executive Nadeem Anjarwalla, detained in Nigeria alongside Mr Gambarayan in February, escaped custody in March.

Zhao's sentencing comes shortly after Sam Bankman-Fried was sentenced to 25 years in prison for fraud committed at his rival crypto platform, FTX.

Widely known as the "crypto king", Bankman-Fried was found to have stolen billions from customers ahead of the firm's failure.

The Justice Department said its investigation into Binance also found the exchange made it easy for criminals to move money.'''
]

print("--- Running Inference on New Sentences ---")

start = time.time()
for text in adverse_media_texts:
    print(f"\nText: {text}")
    entities = ner_pipeline(text)
    print("Entities Found:")
    for entity in entities:
        print(f"  -> Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']:.4f}")
end = time.time()
print("Time required:", end - start)
print("\n--- PROCESS COMPLETE ---")