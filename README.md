# Hybrid_prompt_detection_and-mitigation
Prompt injection attacks threaten LLMs via hidden instructions in user inputs, often bypassing safety constraints. Current defenses, focused on text-based prompts, are vulnerable to injection using multilingual, encoded, or cross-modal inputs such as images and speech. This work proposes a unified defense framework that first standardizes diverse inputs into text. It uses a hybrid reasoning pipeline: a lightweight Chain-of-Draft (CoD) mechanism performs initial risk screening. A hybrid classifier evaluates CoD signals to determine if deeper Chain-of-Thought (CoT) semantic analysis is needed. Suspicious inputs are escalated for detailed analysis, while benign one pass. Malicious prompts are blocked pre-LLM execution, enhancing the security of real-world LLM applications.

Objective: 
To design and implement a robust and efficient prompt injection defense framework for Large Language Models by integrating input normalization for multi-modal, multilingual, encoded, image and speech based inputs followed by a hybrid detection mechanism that combines Chain-of-Draft and Chain-of-Thought reasoning. The framework aims to accurately classify inputs and automatically block malicious prompts to ensure secure and reliable model interaction.

Input:
Text-based user prompts, Multilingual inputs, Encoded or obfuscated text (e.g., Base64, special character encoding), Image inputs, Speech inputs.

Output:
● Classification of the input as Safe or Malicious
● Automaticblocking of malicious prompts, validated prompt forwarded to the LLM (if safe)

<img width="897" height="296" alt="image" src="https://github.com/user-attachments/assets/f0492698-f658-42d8-a48d-8cdc7e69bb0b" />

<img width="904" height="403" alt="image" src="https://github.com/user-attachments/assets/df412f66-415d-4fe4-8f07-f72e18b959ea" />

<img width="888" height="403" alt="image" src="https://github.com/user-attachments/assets/64fcf05d-73ba-4398-800b-d7d6e17761c0" />

<img width="901" height="416" alt="image" src="https://github.com/user-attachments/assets/9f2425c5-88a9-4cf4-8600-62f2225ce3f1" />
