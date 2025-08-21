## 0.1.0 (2025-08-21)

### 🚀 Features

- Create extensible RAG system with llama-powered CLI experience ([a526b73](https://github.com/llama-farm/llamafarm/commit/a526b73))
- comprehensive prompt management system with CLI, testing, and modern LLM support ([#16](https://github.com/llama-farm/llamafarm/pull/16))
- comprehensive models system with real API integration and enhanced CLI ([#15](https://github.com/llama-farm/llamafarm/pull/15))
- add documentation site ([#23](https://github.com/llama-farm/llamafarm/pull/23))
- add home page, chat page w/ dashboard and data section ([#28](https://github.com/llama-farm/llamafarm/pull/28))
- light mode ([#31](https://github.com/llama-farm/llamafarm/pull/31))
- add light mode styling to the prompt and dashboard page ([#48](https://github.com/llama-farm/llamafarm/pull/48))
- Schema-driven project APIs with robust error handling ([#57](https://github.com/llama-farm/llamafarm/pull/57))
- add minimal prompts and runtime support ([#86](https://github.com/llama-farm/llamafarm/pull/86))
- **api:** add initial directory structure ([#11](https://github.com/llama-farm/llamafarm/pull/11))
- **chat:** enable atomic tools ([#40](https://github.com/llama-farm/llamafarm/pull/40))
- **cli:** project initialization ([1b7ed3c](https://github.com/llama-farm/llamafarm/commit/1b7ed3c))
- **cli:** config generator ([bd2f2cf](https://github.com/llama-farm/llamafarm/commit/bd2f2cf))
- **cli:** installer ([#37](https://github.com/llama-farm/llamafarm/pull/37))
- **cli:** add project chat interface ([#50](https://github.com/llama-farm/llamafarm/pull/50))
- **cli:** support dataset operations ([#58](https://github.com/llama-farm/llamafarm/pull/58))
- **cli:** auto-start server + invoke runtime endpoint ([#75](https://github.com/llama-farm/llamafarm/pull/75))
- **config:** generate types and parse configs ([#9](https://github.com/llama-farm/llamafarm/pull/9))
- **config:** support writing config to disk ([48fa185](https://github.com/llama-farm/llamafarm/commit/48fa185))
- **config:** add initial datasets schema ([0d29d0a](https://github.com/llama-farm/llamafarm/commit/0d29d0a))
- **config:** support writing initial config ([f3cb41d](https://github.com/llama-farm/llamafarm/commit/f3cb41d))
- **config:** automate config type generation with datamodel-code-gen… ([#47](https://github.com/llama-farm/llamafarm/pull/47))
- **config:** use dynamic reference to rag schema ([#54](https://github.com/llama-farm/llamafarm/pull/54))
- **core:** add environment config ([479004b](https://github.com/llama-farm/llamafarm/commit/479004b))
- **designer:** add project scaffolding ([2eea644](https://github.com/llama-farm/llamafarm/commit/2eea644))
- **docs:** deploy to docs.llamafarm.dev ([#61](https://github.com/llama-farm/llamafarm/pull/61))
- **rag:** universal retrieval strategies system ([#8](https://github.com/llama-farm/llamafarm/pull/8))
- **rag:** add comprehensive RAG system with strategy-based configuration ([#41](https://github.com/llama-farm/llamafarm/pull/41))
- **runtime:** add placeholder for runtime ([d30a5f2](https://github.com/llama-farm/llamafarm/commit/d30a5f2))
- **server:** add server scaffolding ([748f63f](https://github.com/llama-farm/llamafarm/commit/748f63f))
- **server:** update functions in project service; add data service ([0b127aa](https://github.com/llama-farm/llamafarm/commit/0b127aa))
- **server:** implement rag subsytem into server apis ([#67](https://github.com/llama-farm/llamafarm/pull/67))
- **server/api:** add datasets managements apis ([#26](https://github.com/llama-farm/llamafarm/pull/26))
- **server/api:** add data ingestion and removal apis ([#38](https://github.com/llama-farm/llamafarm/pull/38))
- **services:** add project service ([a17d477](https://github.com/llama-farm/llamafarm/commit/a17d477))

### 🩹 Fixes

- address code review feedback and refactor components ([f071481](https://github.com/llama-farm/llamafarm/commit/f071481))
- **ci:** trivy scan and upload ([#66](https://github.com/llama-farm/llamafarm/pull/66))
- **cli:** config init ([#89](https://github.com/llama-farm/llamafarm/pull/89))
- **config:** address broken type generation ([4375be6](https://github.com/llama-farm/llamafarm/commit/4375be6))
- **config:** test updates ([9221fda](https://github.com/llama-farm/llamafarm/commit/9221fda))
- **docker:** server image build ([#29](https://github.com/llama-farm/llamafarm/pull/29))
- **docs:** include gh actions in deploy package ([#80](https://github.com/llama-farm/llamafarm/pull/80))
- **model:** address bug with strategy parsing ([#83](https://github.com/llama-farm/llamafarm/pull/83))
- **models:** Update demos to work with array-based strategy format ([#74](https://github.com/llama-farm/llamafarm/pull/74))
- **server:** use correct config functions ([83e1213](https://github.com/llama-farm/llamafarm/commit/83e1213))
- **server:** fix self reference in project service ([1828047](https://github.com/llama-farm/llamafarm/commit/1828047))
- **server:** importing of shared config module ([4f6fad7](https://github.com/llama-farm/llamafarm/commit/4f6fad7))

### ❤️ Thank You

- Bobby Radford @BobbyRadford
- Davon Davis @davon-davis
- Matt Hamann
- Racheal Ochalek @rachmlenig
- rachradulo @rachradulo
- rgthelen @rgthelen
- Rob Thelen @rgthelen