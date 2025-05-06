modeller som verkar funka någorlunda:

**Proprietary:**

openai/gpt-4o-2024-11-20

anthropic/claude-3.7-sonnet

google/gemini-2.5-flash-preview

amazon/nova-pro-v1

**Open source:**

**qwen/qwen-2.5-72b-instruct** <-- Funkar bra

qwen/qwen-turbo <--     "content": "Error: 'NoneType' object is not subscriptable",
 
qwen/qwen-plus <- "content": "Error: 'NoneType' object is not subscriptable",

qwen/qwen-max <- inga tool calls

**meta-llama/llama-3.3-70b-instruct** <-- Funkar bra

**meta-llama/llama-3.1-8b-instruct** <-- Tool calls funkar, dåligt svar

x-ai/grok-2-1212 <--    "content": "Error: 'NoneType' object is not subscriptable"

x-ai/grok-beta <-- "content": "Error: 'NoneType' object is not subscriptable"

deepseek/deepseek-r1-distill-llama-70b <- 'NoneType' object is not subscriptable",

nvidia/llama-3.1-nemotron-70b-instruct <-- Tool calls funkar, dåligt svar

**mistralai/ministral-8b**<- tool calls funkar, väldigt inkonsekventa svar

mistralai/mistral-large-2411 <-- försöker anropa funktioner som inte finns

mistralai/mistral-large-2407 <-- funkar dåligt

mistralai/pixtral-large-2411 <-- gör bort sig, Error processing query: 'NoneType' object is not subscriptable

cohere/command-r-plus-08-2024 <- Ger ett rimligt svar men verkar inte exekvera nån sql

ai21/jamba-1-5-large <- Funkar dåligt med tool calls

ai21/jamba-1.6-large <- funkar dåligt med tool calls

nousresearch/hermes-3-llama-3.1-70b <- ger felkod api 500

microsoft/phi-3.5-mini-128k-instruct <-- nonetype not subscriptable

mistralai/mistral-tiny <- funkar dåligt med mer avancerad sql


