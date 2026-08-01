# LangChain Multi-Branch & Parallel Chatbot Application

A Production-Grade Streamlit Chatbot application built using **LangChain Expression Language (LCEL)**, **Pydantic Validation**, and **Groq LLM**.

## 🌟 Core Features

- **Dynamic Routing (`RunnableBranch`)**: Automatically routes queries to specialized prompts (Programming Assistant, Math Tutor, or General Assistant).
- **Parallel Outputs (`RunnableParallel`)**: Concurrently generates responses along with additional metadata (e.g., Summary, Keywords).
- **Structured Output (`Pydantic`)**: Validates and structures LLM responses cleanly.
- **Streamlit UI**: Intuitive and interactive web application.

## 🛠️ Project Structure

```text
langchain-branch-parallel-chatbot/
│
├── app/
│   ├── core/          # Configuration & Logging
│   ├── schemas/       # Request & Response Schemas
│   ├── chatbot/       # Chains, Prompts & Session Memory
│   ├── services/      # Chat Orchestrator Service
│   ├── utils/         # Helper functions
│   └── main.py        # Streamlit App Entrypoint
│
├── tests/             # Unit and integration tests
├── Dockerfile         # Docker build setup
├── docker-compose.yml # Container orchestration
└── requirements.txt   # Project dependencies