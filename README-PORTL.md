# Overview


## Samplers

We have implemented sampling interfaces for the following language model APIs:

- OpenAI: https://platform.openai.com/docs/overview
- Claude: https://www.anthropic.com/api

Make sure to set the `*_API_KEY` environment variables before using these APIs.

## Portl Note
Make sure you set both the `PORTL_API_URL` and the `PORTL_API_KEY` environment variables.
! SET THE OPENAI_API_KEY Env var! 

### Setup 
1. Git clone this repo
```bash 
cd simle-evals-portl
```
```bash
python3 -m venv venv && source venv/bin/activate
```
```bash
cd ../
```

For [HumanEval](https://github.com/openai/human-eval/) (python programming) [use the acuchat forked version which works better with newer python versions]
```bash
git clone https://github.com/AcuChat/human-eval
pip install -e human-eval
```

Now we can install all the packages needed
```bash 
cd simple-evals-portl
```

All deps except for HumanEval are now handled by requirements.txt
```bash 
pip install -r requirements.txt
```

## Running the evals

To run the portl-cascade endpoint on simpleqa, use the following command: 
```bash 
python -m simple-evals-portl.simple_evals --model cascade-portl --eval simpleqa --examples <num_examples>
```

