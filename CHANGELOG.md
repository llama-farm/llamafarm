# Changelog

## [0.0.3](https://github.com/llama-farm/llamafarm/compare/v0.0.2...v0.0.3) (2025-09-11)


### Features

* add a server health check and use it in the CLI ([#133](https://github.com/llama-farm/llamafarm/issues/133)) ([f842ad9](https://github.com/llama-farm/llamafarm/commit/f842ad9c3d661e7765c1063e115b76af46af6bac))
* **cli:** determine server image tag with overrides ([#134](https://github.com/llama-farm/llamafarm/issues/134)) ([2423cfb](https://github.com/llama-farm/llamafarm/commit/2423cfb3ed3b157ba728c0b77bcc38ec5da76f48))
* **designer:** consistent package button and code switcher ([e9c1378](https://github.com/llama-farm/llamafarm/commit/e9c1378a097a20d8e68842a5db00072d6dd31ee8))
* **designer:** package experience inital modal and loader ([79b8298](https://github.com/llama-farm/llamafarm/commit/79b82988c74571002dc8924e9318458d98b98c19))
* **designer:** packaging experience ([e205df2](https://github.com/llama-farm/llamafarm/commit/e205df227718c0b1e34d871f7325cdfeea2dd508))
* **designer:** packet exp loading state, and success states ([0a02614](https://github.com/llama-farm/llamafarm/commit/0a02614bdd90134b47bdb3eafc5a7e586851b81d))
* **designer:** versions page and actions (and dashboard links fixeed) ([e1a1902](https://github.com/llama-farm/llamafarm/commit/e1a1902cf2a19dfcc45ba327a6aaf3db996b1fe6))


### Bug Fixes

* **designer:** config editor switcher not working on some pages ([1d57b1e](https://github.com/llama-farm/llamafarm/commit/1d57b1e7ed7b091246522fc23cb52e8d9ac93cb0))
* **designer:** package expereince run in background and other fixes ([3f8ad9c](https://github.com/llama-farm/llamafarm/commit/3f8ad9c13de1832c4e37bba923992c6a19bb992d))
* **designer:** remove unused ModeToggle import in Data.tsx (TS6133) ([61a0182](https://github.com/llama-farm/llamafarm/commit/61a0182c21235c295f3bc489e490ae911d1160cd))
* **install:** cli download naming ([8649357](https://github.com/llama-farm/llamafarm/commit/864935719201c32c40e5811386addf296288cd48))


### Miscellaneous Chores

* release 0.0.3 ([14e602b](https://github.com/llama-farm/llamafarm/commit/14e602b1286b480b493ec8061d00c831c0871f2a))

## [0.0.2](https://github.com/llama-farm/llamafarm/compare/v0.0.1...v0.0.2) (2025-09-08)


### ⚠ BREAKING CHANGES

* Changed schema from oneOf to anyOf for extractor configs to allow more flexible configurations

### Features

* add comprehensive schema updates and universal strategies system ([#115](https://github.com/llama-farm/llamafarm/issues/115)) ([2359242](https://github.com/llama-farm/llamafarm/commit/2359242cf43e348b037555973ec84bb4b93a0f2b))
* **cli:** add ollama arg and fix docker bugs ([#124](https://github.com/llama-farm/llamafarm/issues/124)) ([1282dd3](https://github.com/llama-farm/llamafarm/commit/1282dd39f4084a5b039a16c59584952fcce4168f))
* **config:** update prompts schema to use role and content ([ba735d1](https://github.com/llama-farm/llamafarm/commit/ba735d1199b234b13d29f5d4c6b90695d2cf0a4a))
* **designer:** add overflow menu, hide cards on dataset view ([d02be98](https://github.com/llama-farm/llamafarm/commit/d02be98cb0ed491a46444c07dbf4c93050bb5f97))
* **designer:** Add RAG tab for processing strategies  ([#116](https://github.com/llama-farm/llamafarm/issues/116)) ([16c6f04](https://github.com/llama-farm/llamafarm/commit/16c6f0423b04f40e601f998ff564b5bb22f39182))
* **designer:** change RAG strategy for a dataset ([32a6a25](https://github.com/llama-farm/llamafarm/commit/32a6a25785936834db35a03a5e757c2e044c96a2))
* **designer:** change rag strategy from dataset page ([2c95e25](https://github.com/llama-farm/llamafarm/commit/2c95e25e21e02ddb6c6da76228cb69dba5fa103a))
* **designer:** connect chat component to api ([#98](https://github.com/llama-farm/llamafarm/issues/98)) ([d697070](https://github.com/llama-farm/llamafarm/commit/d697070c1075b9390c660977a60270481963aac8))
* **designer:** connect data component to api ([#110](https://github.com/llama-farm/llamafarm/issues/110)) ([fe9b9b8](https://github.com/llama-farm/llamafarm/commit/fe9b9b8564e764ce6b9de91a9b7759121a309d27))
* **designer:** create new rag strategy, duplicate strat, update description ([6749c42](https://github.com/llama-farm/llamafarm/commit/6749c42d70ee7fad25b418b18e51e5ea0ec7bcba))
* **designer:** datasets page ([4429ff1](https://github.com/llama-farm/llamafarm/commit/4429ff15dfbd44eaba42d6333c12d24a88219d18))
* **designer:** edit rag strategy, edit, delete, rename (reset button for now) ([121db2b](https://github.com/llama-farm/llamafarm/commit/121db2bad6d6344ded15e35e272ecab620db7d8f))
* **designer:** project api integration ([#107](https://github.com/llama-farm/llamafarm/issues/107)) ([de7c101](https://github.com/llama-farm/llamafarm/commit/de7c101d721c3fd61423c600cdc0577331a3c357))
* **server:** add celery as a task manager for processing entire data… ([#108](https://github.com/llama-farm/llamafarm/issues/108)) ([d9231d6](https://github.com/llama-farm/llamafarm/commit/d9231d62645304e35719e1b353b8a13cc44c0fbd))
* **server:** segment ollama for designer and runtime; handle new prompts ([b66eaae](https://github.com/llama-farm/llamafarm/commit/b66eaaef255156f3cc245851a233364c003bc72e))
* **server:** use instructor_mode from runtime config with openai ([af2930b](https://github.com/llama-farm/llamafarm/commit/af2930b378b3840c2ef819860d235ced995e6b78))
* use llamafarm to build llamafarm ([#112](https://github.com/llama-farm/llamafarm/issues/112)) ([dda49a1](https://github.com/llama-farm/llamafarm/commit/dda49a10097f629e0caef148ecf2d1dcb4d7c901))
* use shared directory for all projects ([#105](https://github.com/llama-farm/llamafarm/issues/105)) ([e813f72](https://github.com/llama-farm/llamafarm/commit/e813f72472a256fba9849f7067e9bb8af6b7ae32))


### Bug Fixes

* **chat:** streaming responses ([#99](https://github.com/llama-farm/llamafarm/issues/99)) ([c2e92cf](https://github.com/llama-farm/llamafarm/commit/c2e92cfa4c4cf135774af760d1bb3230d85af115))
* **cli:** check for ollama running with the correct URL ([e098d87](https://github.com/llama-farm/llamafarm/commit/e098d87ad4639026bf0c75f65d77009dbd66f702))
* **cli:** ensure server is running for projects chat cmd ([748ad61](https://github.com/llama-farm/llamafarm/commit/748ad6150970e1ec3e6ecbf596fde5285ec8c5c2))
* **config:** revert  removal ([a1a1a94](https://github.com/llama-farm/llamafarm/commit/a1a1a94c080c49c6a420ad0964ecd814060b6607))
* **config:** update tests to use new prompts schema ([4937cbe](https://github.com/llama-farm/llamafarm/commit/4937cbee66bba01e0eae088e8ce19ebebce0c885))
* **designer:** add shadcn components and functionality to the 'prompt' page ([#102](https://github.com/llama-farm/llamafarm/issues/102)) ([014f583](https://github.com/llama-farm/llamafarm/commit/014f583f6670e38f09e49ab886469eefbb93c725))
* **designer:** addressing sourcery comments ([c8c51d7](https://github.com/llama-farm/llamafarm/commit/c8c51d7dcc27f2c36b067d8834c4cec343dcd4c6))
* **designer:** build error fix ([cea7da3](https://github.com/llama-farm/llamafarm/commit/cea7da3857256ec8871f27921034c18a0330d9cb))
* **designer:** build issue ([6fd2785](https://github.com/llama-farm/llamafarm/commit/6fd2785a6a5e51e9ee554d5ea48bdb4da053d7a1))
* **designer:** gitignore stuff ([ea063f6](https://github.com/llama-farm/llamafarm/commit/ea063f6bec733155631d828bef93da6d77bec056))
* **designer:** remove 'password updated' notificaiton ([a238524](https://github.com/llama-farm/llamafarm/commit/a2385244598c2c76bcb4321496c07bd2bdf7430b))
* **designer:** remove auto-complete from API secret field in model change ([d43f8e3](https://github.com/llama-farm/llamafarm/commit/d43f8e32ca05822cebec7fa2d4d538d6d5bb5a73))
* **designer:** sourcery comments ([ee1c56e](https://github.com/llama-farm/llamafarm/commit/ee1c56e98c50091c9698325dd8f35e2395e6aba2))
* **designer:** sourcery comments and updates ([9761801](https://github.com/llama-farm/llamafarm/commit/9761801efe377ff1aa20e35a3d4320cb8d9128a8))
* **designer:** sourecery updates pt 2 ([7bb9994](https://github.com/llama-farm/llamafarm/commit/7bb9994c47ff5124283bcee85dc16a744ddfadb3))
* **server:** fix celery logging configuration ([#113](https://github.com/llama-farm/llamafarm/issues/113)) ([5827dc9](https://github.com/llama-farm/llamafarm/commit/5827dc9857e18e2fbf6e46101b7679fc6937049f))
* **server:** remove ollama_host segmentation ([31cc859](https://github.com/llama-farm/llamafarm/commit/31cc859bd45f0c9af81d94a14e4e312fb6430f72))
* **server:** update tests to use new prompts schema ([e7c277e](https://github.com/llama-farm/llamafarm/commit/e7c277eebc8f9e7fbed80d69aadfb2ee6d79ea1b))


### Miscellaneous Chores

* release 0.0.2 ([e2d1aee](https://github.com/llama-farm/llamafarm/commit/e2d1aee7af0bad10bdf4fbb7ce0a102aef22faa3))

## 0.0.1 (2025-08-21)


### Features

* add documentation site ([#23](https://github.com/llama-farm/llamafarm/issues/23)) ([e9146fa](https://github.com/llama-farm/llamafarm/commit/e9146fac560a2d4195de3e137f09102a99ea880f))
* add home page, chat page w/ dashboard and data section ([#28](https://github.com/llama-farm/llamafarm/issues/28)) ([e2afe6a](https://github.com/llama-farm/llamafarm/commit/e2afe6a9b4c1d78237447ba7f434078eb7cc9b14))
* add light mode styling to the prompt and dashboard page ([#48](https://github.com/llama-farm/llamafarm/issues/48)) ([b32adb4](https://github.com/llama-farm/llamafarm/commit/b32adb40b227f3e8c5429d9ee8640df6ce85f269))
* add minimal prompts and runtime support ([#86](https://github.com/llama-farm/llamafarm/issues/86)) ([b2272f9](https://github.com/llama-farm/llamafarm/commit/b2272f98a8f399ee802369c80b9b0c0d36dac9d0))
* **api:** add initial directory structure ([#11](https://github.com/llama-farm/llamafarm/issues/11)) ([65bed86](https://github.com/llama-farm/llamafarm/commit/65bed866a23a5646411526168c7e88d2b611d73f))
* **chat:** enable atomic tools ([#40](https://github.com/llama-farm/llamafarm/issues/40)) ([0845ac0](https://github.com/llama-farm/llamafarm/commit/0845ac03cec636782eef82e676018ca66be91064))
* **cli:** add project chat interface ([#50](https://github.com/llama-farm/llamafarm/issues/50)) ([943cc8c](https://github.com/llama-farm/llamafarm/commit/943cc8c752199175e75ff13709621fab3e78d8ad))
* **cli:** auto-start server + invoke runtime endpoint ([#75](https://github.com/llama-farm/llamafarm/issues/75)) ([bc18010](https://github.com/llama-farm/llamafarm/commit/bc180106ee81d4680843367be06e6c353785ea20))
* **cli:** config generator ([bd2f2cf](https://github.com/llama-farm/llamafarm/commit/bd2f2cfcdf07efd56b7f7cb14f7127d5dc52e2b6))
* **cli:** installer ([#37](https://github.com/llama-farm/llamafarm/issues/37)) ([b57f6f8](https://github.com/llama-farm/llamafarm/commit/b57f6f8740adac88c3ea652ad7439b60b0f75094))
* **cli:** project initialization ([1b7ed3c](https://github.com/llama-farm/llamafarm/commit/1b7ed3c869fb7d251e4dde2dddccb3085a1c1fdd))
* **cli:** support dataset operations ([#58](https://github.com/llama-farm/llamafarm/issues/58)) ([fe40ef3](https://github.com/llama-farm/llamafarm/commit/fe40ef3bdfc07616e8eade7d3cba6a5ffd5bda09))
* comprehensive models system with real API integration and enhanced CLI ([#15](https://github.com/llama-farm/llamafarm/issues/15)) ([1c484a4](https://github.com/llama-farm/llamafarm/commit/1c484a4ff3e7523731ce6065eae29beb4d10ea33))
* comprehensive prompt management system with CLI, testing, and modern LLM support ([#16](https://github.com/llama-farm/llamafarm/issues/16)) ([85be2ea](https://github.com/llama-farm/llamafarm/commit/85be2ea96ed4243db407a7aede6d366086595886))
* **config:** add initial datasets schema ([0d29d0a](https://github.com/llama-farm/llamafarm/commit/0d29d0a68e72d673d65b5651a4491f9073e78cc5))
* **config:** automate config type generation with datamodel-code-gen… ([#47](https://github.com/llama-farm/llamafarm/issues/47)) ([bc774e6](https://github.com/llama-farm/llamafarm/commit/bc774e61f35ed9854e00d807f2863de69e5de77d))
* **config:** generate types and parse configs ([#9](https://github.com/llama-farm/llamafarm/issues/9)) ([011b425](https://github.com/llama-farm/llamafarm/commit/011b4251e45a34d6408d089fa0a73174c9486103))
* **config:** support writing config to disk ([48fa185](https://github.com/llama-farm/llamafarm/commit/48fa18507a5eff8d418f9e9e1a7c7ee015c23c20))
* **config:** support writing initial config ([804ba26](https://github.com/llama-farm/llamafarm/commit/804ba26f45fe8655e8d6b9a8eb7f8beee39f5a6c))
* **config:** support writing initial config ([f3cb41d](https://github.com/llama-farm/llamafarm/commit/f3cb41daa4cbe10d106452a9d14647faf0ea5190))
* **config:** use dynamic reference to rag schema ([#54](https://github.com/llama-farm/llamafarm/issues/54)) ([a39c5e6](https://github.com/llama-farm/llamafarm/commit/a39c5e60c9633c552457a69a2e0fc1dbbccbb912))
* **core:** add environment config ([479004b](https://github.com/llama-farm/llamafarm/commit/479004b595e7a9ea1996e6fbc91258c433b47f3b))
* dataset file search/additional light mode adjustments ([1e25251](https://github.com/llama-farm/llamafarm/commit/1e2525188dafc941efe500710aad99ee08153154))
* **designer:** add project scaffolding ([2eea644](https://github.com/llama-farm/llamafarm/commit/2eea644e22f6b7c14767ac17cd81779b4d52a6b5))
* **docs:** deploy to docs.llamafarm.dev ([#61](https://github.com/llama-farm/llamafarm/issues/61)) ([772e334](https://github.com/llama-farm/llamafarm/commit/772e334237d0baa34dc7844c267dc43ea1f50456))
* light mode ([#31](https://github.com/llama-farm/llamafarm/issues/31)) ([da2cca0](https://github.com/llama-farm/llamafarm/commit/da2cca05b8ad75bde6672c5ccbabee1db12bd198))
* **models:** add core model management system ([#63](https://github.com/llama-farm/llamafarm/issues/63)) ([3e88fb8](https://github.com/llama-farm/llamafarm/commit/3e88fb8bc4cf3cdf2127199903862dc3df0e678f))
* **rag:** add comprehensive RAG system with strategy-based configuration ([#41](https://github.com/llama-farm/llamafarm/issues/41)) ([3b52e00](https://github.com/llama-farm/llamafarm/commit/3b52e00a9848b9899748c99b16d2f4a07d9f998a))
* **rag:** universal retrieval strategies system ([#8](https://github.com/llama-farm/llamafarm/issues/8)) ([7ad49f4](https://github.com/llama-farm/llamafarm/commit/7ad49f48ee9e12071487355f7892b7b82409829f))
* **runtime:** add placeholder for runtime ([d30a5f2](https://github.com/llama-farm/llamafarm/commit/d30a5f24956999122c428f79299ae1907b282fa2))
* Schema-driven project APIs with robust error handling ([#57](https://github.com/llama-farm/llamafarm/issues/57)) ([1d9a385](https://github.com/llama-farm/llamafarm/commit/1d9a385cfc6872bda1316207f07b74b75f0bbfbb))
* **server/api:** add data ingestion and removal apis ([#38](https://github.com/llama-farm/llamafarm/issues/38)) ([dacc7c9](https://github.com/llama-farm/llamafarm/commit/dacc7c9cee8a77c93fa342ca3700473580eabe6a))
* **server/api:** add datasets managements apis ([#26](https://github.com/llama-farm/llamafarm/issues/26)) ([1c96bd2](https://github.com/llama-farm/llamafarm/commit/1c96bd21d1610453234e49528c138440218b74df))
* **server:** add server scaffolding ([748f63f](https://github.com/llama-farm/llamafarm/commit/748f63fc09eb5162d6efbf1b74b4cbeff74e6e71))
* **server:** implement rag subsytem into server apis ([#67](https://github.com/llama-farm/llamafarm/issues/67)) ([d97b58f](https://github.com/llama-farm/llamafarm/commit/d97b58f4f7563231efef2f85d9738f0cc173ba50))
* **server:** update functions in project service; add data service ([0b127aa](https://github.com/llama-farm/llamafarm/commit/0b127aac5f77908cfc054a044a2bede3855d0aa3))
* **services:** add project service ([a17d477](https://github.com/llama-farm/llamafarm/commit/a17d4778c4fa17f8e30ae08fc7ee99670aacfa6b))


### Bug Fixes

* address code review feedback and refactor components ([f071481](https://github.com/llama-farm/llamafarm/commit/f071481d73996c24dc37a2cfc841282b35b2201f))
* **ci:** trivy scan and upload ([#66](https://github.com/llama-farm/llamafarm/issues/66)) ([7e70fb0](https://github.com/llama-farm/llamafarm/commit/7e70fb070fddf971965a5fea6cc182fc1ce1094b))
* **cli:** config init ([#89](https://github.com/llama-farm/llamafarm/issues/89)) ([2cfd764](https://github.com/llama-farm/llamafarm/commit/2cfd764da49bc95180f0c329fd41a5ac7674e53b))
* **config:** address broken type generation ([4375be6](https://github.com/llama-farm/llamafarm/commit/4375be646f09acbe9e3daa66fb92eb78fa035b31))
* **config:** test updates ([cc6a0a6](https://github.com/llama-farm/llamafarm/commit/cc6a0a66fec7166b20e05706fcdebd60d614494c))
* **config:** test updates ([9221fda](https://github.com/llama-farm/llamafarm/commit/9221fda65412c714876f6b8aee34fc8d7b702f87))
* **docker:** server image build ([#29](https://github.com/llama-farm/llamafarm/issues/29)) ([9e3b726](https://github.com/llama-farm/llamafarm/commit/9e3b726d4dd5be809fb06e467e596001b4a3f436))
* **docs:** include gh actions in deploy package ([#80](https://github.com/llama-farm/llamafarm/issues/80)) ([568cbd2](https://github.com/llama-farm/llamafarm/commit/568cbd256d023968638a7d76827e8a7e58e5a1f0))
* **model:** address bug with strategy parsing ([#83](https://github.com/llama-farm/llamafarm/issues/83)) ([94ddecd](https://github.com/llama-farm/llamafarm/commit/94ddecd3f903040c914a18c72335f83c4e85a50b))
* **models:** Update demos to work with array-based strategy format ([#74](https://github.com/llama-farm/llamafarm/issues/74)) ([eab2139](https://github.com/llama-farm/llamafarm/commit/eab213935ce067e8441df3e1e178a88ee7ac97cc))
* **server:** fix self reference in project service ([1828047](https://github.com/llama-farm/llamafarm/commit/18280478307fe795bd807312ee6de8575a5a2b82))
* **server:** importing of shared config module ([4f6fad7](https://github.com/llama-farm/llamafarm/commit/4f6fad7a09179f5f04bb35682d7a102bf76cc806))
* **server:** use correct config functions ([83e1213](https://github.com/llama-farm/llamafarm/commit/83e1213a0ef0bb6f094f6b4ce561a6087eedd594))


### Miscellaneous Chores

* release 0.0.1 ([1517073](https://github.com/llama-farm/llamafarm/commit/1517073440afe0054407e46362794c3316cc579d))

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
