# Changelog

## [0.0.17](https://github.com/llama-farm/llamafarm/compare/v0.0.16...v0.0.17) (2025-11-24)


### Bug Fixes

* **designer:** fix empty prompts array for new projects, fix current version in designer ([#495](https://github.com/llama-farm/llamafarm/issues/495)) ([488313c](https://github.com/llama-farm/llamafarm/commit/488313cd239c2075f2adcce47b297e0b89005494))
* **docs:** add troubleshooting and update styles ([#498](https://github.com/llama-farm/llamafarm/issues/498)) ([270af60](https://github.com/llama-farm/llamafarm/commit/270af6050c3007744aa1e8ea778afe6ce3f66177))
* hf progress bars can crash runtime ([#499](https://github.com/llama-farm/llamafarm/issues/499)) ([50d66a8](https://github.com/llama-farm/llamafarm/commit/50d66a847062b5071e47b66231e1b9e9297f2e88))

## [0.0.16](https://github.com/llama-farm/llamafarm/compare/v0.0.15...v0.0.16) (2025-11-23)


### Bug Fixes

* cli packaging ([8058b1f](https://github.com/llama-farm/llamafarm/commit/8058b1f3f8c809e5896a47e3d73a579d8a14fbfd))

## [0.0.15](https://github.com/llama-farm/llamafarm/compare/v0.0.14...v0.0.15) (2025-11-22)


### Features

* Add universal event logging for observability ([#383](https://github.com/llama-farm/llamafarm/issues/383)) ([e0c4c8e](https://github.com/llama-farm/llamafarm/commit/e0c4c8e89582a8ab117f28f5cd9af3fbffd6080c))
* create project from existing project ([#459](https://github.com/llama-farm/llamafarm/issues/459)) ([3fee20d](https://github.com/llama-farm/llamafarm/commit/3fee20d3801953bd69c687c10c28bb82e925b8d2))
* **designer:** Add interactive demo project creation system ([#473](https://github.com/llama-farm/llamafarm/issues/473)) ([4328380](https://github.com/llama-farm/llamafarm/commit/4328380e589f5a7db95f94769f1eb41afe65711d))
* **designer:** demo app, copy config, test fixes ([#477](https://github.com/llama-farm/llamafarm/issues/477)) ([c840da7](https://github.com/llama-farm/llamafarm/commit/c840da7a29c817baf29ed86bc886b000cf1a466b))
* **designer:** import demo data, demo to universal runtime, fix empty prompts array bug ([#491](https://github.com/llama-farm/llamafarm/issues/491)) ([dc5a33f](https://github.com/llama-farm/llamafarm/commit/dc5a33fc368fc8dbd2d29704be35a441d61f5cf1))
* **electron:** Desktop app with auto-updates and polished UI ([#458](https://github.com/llama-farm/llamafarm/issues/458)) ([02ddf58](https://github.com/llama-farm/llamafarm/commit/02ddf5875fce359f3df7d6d5be6ca822aafb0b01))
* implement enhanced tool calling ([#460](https://github.com/llama-farm/llamafarm/issues/460)) ([72d9b82](https://github.com/llama-farm/llamafarm/commit/72d9b82abee8611d00ee4290556b751736d22992))
* **runtime:** support gguf models ([#445](https://github.com/llama-farm/llamafarm/issues/445)) ([6479372](https://github.com/llama-farm/llamafarm/commit/64793727a30651d2de771410dd8f56a1f8604222))


### Bug Fixes

* Add GH_TOKEN to desktop app workflows ([#480](https://github.com/llama-farm/llamafarm/issues/480)) ([6402da0](https://github.com/llama-farm/llamafarm/commit/6402da075ec2a7a90743ef6956215a8d188d39f8))
* **cli:** prevent uv from building source on run ([#482](https://github.com/llama-farm/llamafarm/issues/482)) ([082a269](https://github.com/llama-farm/llamafarm/commit/082a269673c3342f2bfa535c877d5fcbfb6aeb8b))
* **cli:** sync newer config on startup instead of always preferring CWD ([#484](https://github.com/llama-farm/llamafarm/issues/484)) ([458cd7b](https://github.com/llama-farm/llamafarm/commit/458cd7bfa48655ec92963bb36444ad52e142a22a))
* **cli:** upgrades may fail on unix-like environments ([#453](https://github.com/llama-farm/llamafarm/issues/453)) ([3efed4c](https://github.com/llama-farm/llamafarm/commit/3efed4c31c1f684be9aa3e2078aee41907e06ede))
* **designer:** database tab switching ([#455](https://github.com/llama-farm/llamafarm/issues/455)) ([e8be498](https://github.com/llama-farm/llamafarm/commit/e8be498a96b1adec475460d0ca1eff980ec33a67))
* **designer:** dataset name validation, status column, other polishes ([#472](https://github.com/llama-farm/llamafarm/issues/472)) ([bcc8669](https://github.com/llama-farm/llamafarm/commit/bcc86699c237cc2c95fd1b4a49a1ffc71175e907))
* **designer:** Improve RAG integration and chat context management ([#471](https://github.com/llama-farm/llamafarm/issues/471)) ([7bb1011](https://github.com/llama-farm/llamafarm/commit/7bb1011d13ab16148eb5c3a383f03de8ad19fc9e))
* **mcp:** handle missing mcp config section gracefully ([#485](https://github.com/llama-farm/llamafarm/issues/485)) ([517d663](https://github.com/llama-farm/llamafarm/commit/517d663b303d315a1ace964669fb7d5d94eed5a4))


### Miscellaneous Chores

* release 0.0.15 ([2395ddc](https://github.com/llama-farm/llamafarm/commit/2395ddcc68a71ff500dd6a20dbabe69ba2cdac63))

## [0.0.14](https://github.com/llama-farm/llamafarm/compare/v0.0.13...v0.0.14) (2025-11-13)


### Features

* **cli:** show size stats in raq query result ([#442](https://github.com/llama-farm/llamafarm/issues/442)) ([f602692](https://github.com/llama-farm/llamafarm/commit/f6026924ba03fdc719c16481d4cd385ad9b754e4))
* **designer:** database embedding and retrieval strategies ([#438](https://github.com/llama-farm/llamafarm/issues/438)) ([db5ee11](https://github.com/llama-farm/llamafarm/commit/db5ee110dd840b00f8cb8d476d087d0594b4a0be))


### Bug Fixes

* **designer:** build chat errors ([#444](https://github.com/llama-farm/llamafarm/issues/444)) ([964ed95](https://github.com/llama-farm/llamafarm/commit/964ed95f2268c787a5dbe01aa9025e0bb3049263))
* **designer:** file drop dataset selection ([#423](https://github.com/llama-farm/llamafarm/issues/423)) ([1bd0a42](https://github.com/llama-farm/llamafarm/commit/1bd0a429103b3cc8d3ff69b0ff804862ee92b506))
* **package:** missing ruff config for datamodel gen ([#450](https://github.com/llama-farm/llamafarm/issues/450)) ([1e5a31b](https://github.com/llama-farm/llamafarm/commit/1e5a31bad7bec5a2bf4c8ea4932fffe47896bf06))


### Miscellaneous Chores

* release 0.0.14 ([58522a1](https://github.com/llama-farm/llamafarm/commit/58522a1a681030d1b4328b23da39b06c1acf676d))

## [0.0.13](https://github.com/llama-farm/llamafarm/compare/v0.0.12...v0.0.13) (2025-11-11)


### Bug Fixes

* **cli:** use correct version number ([#440](https://github.com/llama-farm/llamafarm/issues/440)) ([d01f9b8](https://github.com/llama-farm/llamafarm/commit/d01f9b872b20e8e283daafe30a1e313c89682d99))

## [0.0.12](https://github.com/llama-farm/llamafarm/compare/v0.0.11...v0.0.12) (2025-11-11)


### Features

* add tests for project deletion ([afdf3a5](https://github.com/llama-farm/llamafarm/commit/afdf3a51e4b54313ff1345528009c068468052c4))
* **cli:** add delete projects command ([#420](https://github.com/llama-farm/llamafarm/issues/420)) ([0a03d2a](https://github.com/llama-farm/llamafarm/commit/0a03d2adf54a2d3b3bc09ddacfe3ba1b2a502441))
* **designer:** add copy button to config editor ([af22097](https://github.com/llama-farm/llamafarm/commit/af22097401cb951cfd9591d20c59f8270d8f7169))
* **designer:** config anchor points and outline, jump to section, parse config, highligh ([04b3c52](https://github.com/llama-farm/llamafarm/commit/04b3c528a402b391274ef1598416f68dac28dae4))
* **designer:** config unsaved changes modal ([fd8c588](https://github.com/llama-farm/llamafarm/commit/fd8c588cf6f7914a694809b1ad582faac5fbb11f))
* **designer:** scroll to spot (search incl) ([0c9b707](https://github.com/llama-farm/llamafarm/commit/0c9b70769a12b8ba4a17ed8f05add8cfe350cfb1))
* **designer:** search field in config editor ([1a2c2a8](https://github.com/llama-farm/llamafarm/commit/1a2c2a837daece6085058460765acab64560cb89))
* **designer:** tab specific section jumping for config editor toggling ([61edd24](https://github.com/llama-farm/llamafarm/commit/61edd2445724fa6ccf4567455bf8b1b4857eeabd))
* **designer:** unsaved changes prompt for config editor ([401fcf7](https://github.com/llama-farm/llamafarm/commit/401fcf7091a6e1f857f4dab89feec443609da945))
* implement delete_project in ProjectService ([1d896b0](https://github.com/llama-farm/llamafarm/commit/1d896b0c74d600bd9531868dcaeb410722188e68))
* **server:** add embedding strategies to database api ([#428](https://github.com/llama-farm/llamafarm/issues/428)) ([053b965](https://github.com/llama-farm/llamafarm/commit/053b9658ace6d4b624c1ca6dab758527396e3e1a))
* **server:** add mcp_server prop to runtime config ([#415](https://github.com/llama-farm/llamafarm/issues/415)) ([693ecad](https://github.com/llama-farm/llamafarm/commit/693ecadff546bac8d5eae3a8b79ba03a7ba02f0d))
* **server:** add project context provider ([#417](https://github.com/llama-farm/llamafarm/issues/417)) ([4d50045](https://github.com/llama-farm/llamafarm/commit/4d50045841f17fc61c635edcfa43cad565e15e12))
* **server:** implement delete project api ([db0536b](https://github.com/llama-farm/llamafarm/commit/db0536b9035717d932d6827a759eba3fd9deb3c5))
* update API endpoint to use new delete method ([8e6233d](https://github.com/llama-farm/llamafarm/commit/8e6233d2fe2fd7f9e30c30131b5b053cb3686099))


### Bug Fixes

* allow deleting projects with invalid configs ([b125192](https://github.com/llama-farm/llamafarm/commit/b125192f0512f9e4c6ce44fd489df19fbc36a672))
* clean up in-memory sessions ([476fbc2](https://github.com/llama-farm/llamafarm/commit/476fbc2cee52bab853acad87488328e0627e9942))
* **cli:** ensure we use a managed python ([#410](https://github.com/llama-farm/llamafarm/issues/410)) ([13683ea](https://github.com/llama-farm/llamafarm/commit/13683ea810469971e27fea3fb2ee3821f59bc4b8))
* delete entire directory ([8cfc62d](https://github.com/llama-farm/llamafarm/commit/8cfc62d131e606962a93a847395ae6c0f2ff4e0a))
* **designer:** add footer to chat window to nudge testing chats to test tab ([1ba4fd8](https://github.com/llama-farm/llamafarm/commit/1ba4fd8bb21b59603875f6f7bc29aa3cb8b61be0))
* **designer:** add title to chat window "build assistant" ([020080e](https://github.com/llama-farm/llamafarm/commit/020080efc106dec44a8a8126013789b801a948d5))
* **designer:** address comments ([3908d4c](https://github.com/llama-farm/llamafarm/commit/3908d4c59889b1208c823a5827d4771d4b0df8fc))
* **designer:** address comments and slim down PR maybe ([6f465c8](https://github.com/llama-farm/llamafarm/commit/6f465c86b75c70f3ca01ce9a87ee867abcbfce09))
* **designer:** address cubic comment ([6a36728](https://github.com/llama-farm/llamafarm/commit/6a3672878d536a612d7ce7a51d7f95c23c9887ea))
* **designer:** addressing comments ([6f26ae8](https://github.com/llama-farm/llamafarm/commit/6f26ae8f89c3f317292379904bdc0c9118abfe09))
* **designer:** addressing comments ([7f491b6](https://github.com/llama-farm/llamafarm/commit/7f491b625cadd08714a1fb0c75e8fae5ccba3be1))
* **designer:** addressing field validation comment ([fbcf7f5](https://github.com/llama-farm/llamafarm/commit/fbcf7f5c1332ae1ab42a8e919622c24387db15f2))
* **designer:** allow edit database ([b9bddb9](https://github.com/llama-farm/llamafarm/commit/b9bddb98f878a0873270d3b305e1397c60da1230))
* **designer:** allow model swap from existing project and disk models ([#433](https://github.com/llama-farm/llamafarm/issues/433)) ([0e5870a](https://github.com/llama-farm/llamafarm/commit/0e5870af54395f9e42c69956498b1089490a2a54))
* **designer:** allow project delete from designer with new API ([#418](https://github.com/llama-farm/llamafarm/issues/418)) ([ae9c6ba](https://github.com/llama-farm/llamafarm/commit/ae9c6ba7e700d914398b4a798ea3a29f8017eeb5))
* **designer:** bug in config editor with collapsing ad expanding menus ([fd08028](https://github.com/llama-farm/llamafarm/commit/fd08028fede58e9716f7c65d0b83ab610efe6766))
* **designer:** build error end2end fix maybe ([d850c83](https://github.com/llama-farm/llamafarm/commit/d850c8391bb592f07fb32b7ba18a3b4ecd50d28b))
* **designer:** build errors ([73bdfa5](https://github.com/llama-farm/llamafarm/commit/73bdfa511b422cb11fb6cc88e2ccb689d6413983))
* **designer:** chat response scroll ux ([5c027f6](https://github.com/llama-farm/llamafarm/commit/5c027f69e96253faf2ac65bccf3d7e5fe4a5eaf4))
* **designer:** clean up code ([92021af](https://github.com/llama-farm/llamafarm/commit/92021af2d0a29630fe089f497b0736c78e1a36eb))
* **designer:** config editor contents style polish ([938a71b](https://github.com/llama-farm/llamafarm/commit/938a71bc79eae7bfe425e885ba9555db9ffe4721))
* **designer:** data config editor was not loading ([b0ebc8e](https://github.com/llama-farm/llamafarm/commit/b0ebc8e053e19d9a00f6d6fbbed64f2f23d94091))
* **designer:** fix build errors and jump to section issue ([6687e8f](https://github.com/llama-farm/llamafarm/commit/6687e8f7ecc3cab22aedd64e50cf54fd3d2f90dc))
* **designer:** fix issue with unsaved changes modal not appearing ([efca91c](https://github.com/llama-farm/llamafarm/commit/efca91cfc13cd9373e2e9dce9f10365f46ad930d))
* **designer:** fix scroll issue on the botton of the dataset page ([01c4567](https://github.com/llama-farm/llamafarm/commit/01c4567c19d4ef665515763b9d91a2114298809a))
* **designer:** fix tab issue created by last commit ([8191181](https://github.com/llama-farm/llamafarm/commit/81911814b996ee2ff82cc597e9dcb36ab9534e81))
* **designer:** fix test page bug with switcher save ([32bda2e](https://github.com/llama-farm/llamafarm/commit/32bda2e31fe71f62f1536731894fae13bc0e381e))
* **designer:** fixed bug where editing config cleared contents panel state ([24bbf66](https://github.com/llama-farm/llamafarm/commit/24bbf6656dfd32e8d467be2ab7a48b774e5de59b))
* **designer:** fixing what I broke ([a8b5212](https://github.com/llama-farm/llamafarm/commit/a8b5212060e070c47d396e6a2d3cf4eb9b527c99))
* **designer:** mobile mode switch to test tab was not working ([71beea5](https://github.com/llama-farm/llamafarm/commit/71beea52dbdf036fa80459fe46365083cb299009))
* **designer:** new project creation experience ([#411](https://github.com/llama-farm/llamafarm/issues/411)) ([2f9db9d](https://github.com/llama-farm/llamafarm/commit/2f9db9df61e557831f23fc7c200705a34c85f5f0))
* **designer:** re-add "add database" button ([4fcdc3a](https://github.com/llama-farm/llamafarm/commit/4fcdc3a159756dbb57a751da8b880902f74a8cf0))
* **designer:** removing code that wasn't working (scroll) ([fd63c41](https://github.com/llama-farm/llamafarm/commit/fd63c418eadaf8f9b5fe751496e00eea5377b4b7))
* **designer:** simplify create projects fields ([cf20236](https://github.com/llama-farm/llamafarm/commit/cf202361b4f1662efc1d297c58f8aa29c5eb2b15))
* **designer:** simplify PR ([084efb8](https://github.com/llama-farm/llamafarm/commit/084efb888664b0d7c28a3634697d4f7969b1a8f1))
* **designer:** small edits to dropdown ([0ae6405](https://github.com/llama-farm/llamafarm/commit/0ae6405c4e46810cc122f157b0c46b61450e23f0))
* **designer:** tool call and chat style enhancements + markdown edits ([f774e7e](https://github.com/llama-farm/llamafarm/commit/f774e7e2adc4d9d92ba53dd7d673782b8829e5bd))
* **designer:** upgrade show correct version ([#406](https://github.com/llama-farm/llamafarm/issues/406)) ([11cf9cd](https://github.com/llama-farm/llamafarm/commit/11cf9cd24accd0cac30676a10a52039cbeb597a1))


### Miscellaneous Chores

* release 0.0.12 ([dd942a1](https://github.com/llama-farm/llamafarm/commit/dd942a1d59ef303a8914bd50f3eb4a9055dfaeea))

## [0.0.11](https://github.com/llama-farm/llamafarm/compare/v0.0.10...v0.0.11) (2025-10-31)


### Features

* **designer:** run as part of server ([#389](https://github.com/llama-farm/llamafarm/issues/389)) ([72b7942](https://github.com/llama-farm/llamafarm/commit/72b7942711b8b8dfdfa7971a817d9295458d38b3))


### Bug Fixes

* **designer:** make the chats work ([#387](https://github.com/llama-farm/llamafarm/issues/387)) ([f4e3e0b](https://github.com/llama-farm/llamafarm/commit/f4e3e0b33e0adb15acc3ea5a7aac601a1cb99ffb))
* **runtime:** docker tweak ([54ba31a](https://github.com/llama-farm/llamafarm/commit/54ba31a611c9334c2c4dbfee5541c878e87f8665))


### Miscellaneous Chores

* release 0.0.11 ([f624e6b](https://github.com/llama-farm/llamafarm/commit/f624e6bbb460588f8427b90bc24650f6bb12d54e))

## [0.0.10](https://github.com/llama-farm/llamafarm/compare/v0.0.9...v0.0.10) (2025-10-31)


### Features

* add MCP server and improve config importing ([#301](https://github.com/llama-farm/llamafarm/issues/301)) ([65fc50f](https://github.com/llama-farm/llamafarm/commit/65fc50faeef3a34e62d69d946da36d85e8e8a355))
* **cli:** add services status command ([#363](https://github.com/llama-farm/llamafarm/issues/363)) ([69166ce](https://github.com/llama-farm/llamafarm/commit/69166ce3042f00dbda260f70beb44ef4a3698d1a))
* **cli:** enable upgrades via TUI menu ([#339](https://github.com/llama-farm/llamafarm/issues/339)) ([ec80e34](https://github.com/llama-farm/llamafarm/commit/ec80e34bf887d699be25e7b51514dca1a9816ac9))
* **designer:** allow models to refernce prompt sets ([0865e3b](https://github.com/llama-farm/llamafarm/commit/0865e3b6fdc1e8222b7a0d2db60cbd05d856e422))
* **designer:** double click to toggle to tab home state + polish padding on header ([fa30e2a](https://github.com/llama-farm/llamafarm/commit/fa30e2aa19ec5310c11073136b81cc781f3757c5))
* **designer:** download models and use disk models ([#382](https://github.com/llama-farm/llamafarm/issues/382)) ([48c8ee5](https://github.com/llama-farm/llamafarm/commit/48c8ee591b7cb480f1f18dc5df73097d07afd934))
* **designer:** Enhanced dataset processing progress tracking and results display ([#375](https://github.com/llama-farm/llamafarm/issues/375)) ([6689606](https://github.com/llama-farm/llamafarm/commit/668960633fd2d968d5217be8f93a7734ed76475e))
* **designer:** hide cloud models, update local models to capture name and description ([7b9cc80](https://github.com/llama-farm/llamafarm/commit/7b9cc8008a1dbc42858aedbdba73ba4b6130b11b))
* **designer:** hook up dev chat to designer chat ([ab56435](https://github.com/llama-farm/llamafarm/commit/ab56435952faf79b4201ab3dde11059b64d8e8b3))
* **designer:** hook up models to read and display the correct model ([c594db9](https://github.com/llama-farm/llamafarm/commit/c594db93c465d9e2ae6db83992509cb44e9a6b65))
* **designer:** hook up models to read and display the correct model ([8fb249d](https://github.com/llama-farm/llamafarm/commit/8fb249d441dfaa2cc217ef160dc2a616edc4c63f))
* **designer:** hook up test chat ([#378](https://github.com/llama-farm/llamafarm/issues/378)) ([92e4eff](https://github.com/llama-farm/llamafarm/commit/92e4eff3b5702868cb4908c04c6838d01002eda2))
* **designer:** prompts and models ux quick updates so they can be hooked up ([fe47d08](https://github.com/llama-farm/llamafarm/commit/fe47d086905024e0c4c3828e009e3e311879d374))
* **designer:** update "add prompt" modal to "add to set" ([16d4848](https://github.com/llama-farm/llamafarm/commit/16d4848651ebf7953f83a46df371c0aba7e44cf0))
* **designer:** Updating the Data to be dynamic, update config, and process folders.  ([#361](https://github.com/llama-farm/llamafarm/issues/361)) ([bbc69ee](https://github.com/llama-farm/llamafarm/commit/bbc69ee57afd474643aa6ffd64279f113348a823))
* **designer:** UX for prompt sets ([67e3a77](https://github.com/llama-farm/llamafarm/commit/67e3a771d79a50b24a682692023f053981c23cdd))
* editable code in code viewer ([8bf1fe9](https://github.com/llama-farm/llamafarm/commit/8bf1fe9a07d8529c8a4a02ba23fe70fd0bf0509f))
* **rag:** .msg parsing and loader refactor ([#309](https://github.com/llama-farm/llamafarm/issues/309)) ([424eadf](https://github.com/llama-farm/llamafarm/commit/424eadf6da8d6e86aa2010993bc2eeb10bfa201e))
* universal runtime and native processes ([#349](https://github.com/llama-farm/llamafarm/issues/349)) ([d535842](https://github.com/llama-farm/llamafarm/commit/d535842ad0d0ea092de2352711445e46647dc562))


### Bug Fixes

* add Python version constraints for PyTorch compatibility ([#380](https://github.com/llama-farm/llamafarm/issues/380)) ([df0a9b4](https://github.com/llama-farm/llamafarm/commit/df0a9b466c8ed418cdb6b4e6970ed97d6cd9ae3d))
* **ci:** bust Go build cache when source files change ([b1f0c24](https://github.com/llama-farm/llamafarm/commit/b1f0c248c4f4d1d032147c8f4de224c7491ba5fa))
* **ci:** make chmod non-fatal in docker compose test ([2cde502](https://github.com/llama-farm/llamafarm/commit/2cde50297526d1b4198f51405196cf77ff9778d4)), closes [#372](https://github.com/llama-farm/llamafarm/issues/372)
* **ci:** resolve docker compose test permission error on self-hosted runners ([f73e4c6](https://github.com/llama-farm/llamafarm/commit/f73e4c62ed4cf29251fa4184211f7b2dc96ec146)), closes [#367](https://github.com/llama-farm/llamafarm/issues/367)
* **ci:** use sudo to remove docker-created files in test cleanup ([0c3f532](https://github.com/llama-farm/llamafarm/commit/0c3f532c2eb34f1cd45c0fc3c5ca3b6d568fe6ea))
* Clear-text logging of sensitive information ([#345](https://github.com/llama-farm/llamafarm/issues/345)) ([5a0f2be](https://github.com/llama-farm/llamafarm/commit/5a0f2bef130a6792bf2099bba982bc498656ad90))
* **dashboard:** swap out fake metric cards on dashboard with real counts ([f940dd2](https://github.com/llama-farm/llamafarm/commit/f940dd24911d4a1ba76def2afa5989be6aef63af))
* **designer:** add 404 page ([ad95bd4](https://github.com/llama-farm/llamafarm/commit/ad95bd4b91299c04b910c68f0b5069a788a0e768))
* **designer:** add prompts empty state ([c7c1763](https://github.com/llama-farm/llamafarm/commit/c7c176341f872ef8235e8d7358abd4787079329a))
* **designer:** address build errors and things ([84f3fcb](https://github.com/llama-farm/llamafarm/commit/84f3fcb8b9fd9f0cdf83950fe5802acd8737686b))
* **designer:** address code review issues ([74cfdde](https://github.com/llama-farm/llamafarm/commit/74cfddeafb6909a7c19eace9a6ef0ff86fc69909))
* **designer:** addressing comments and simplifying I think ([69163e6](https://github.com/llama-farm/llamafarm/commit/69163e670d84434a75009b16404da5e9d2984af2))
* **designer:** addressing high priority AI comments ([0bd65e2](https://github.com/llama-farm/llamafarm/commit/0bd65e2c4fa90d5cdde34e2f97c0b62cf1f6cd53))
* **designer:** build errors ([d3b1c95](https://github.com/llama-farm/llamafarm/commit/d3b1c9541e317b3dee1e4132494a8e689e957f3d))
* **designer:** build errors ([7ddc1a4](https://github.com/llama-farm/llamafarm/commit/7ddc1a488543d0213ba84120fec9ef6b36caad0b))
* **designer:** build errors ([2176bcf](https://github.com/llama-farm/llamafarm/commit/2176bcf20ee18db88d103c072cad67142dcdfc62))
* **designer:** build errors and comments ([60bb603](https://github.com/llama-farm/llamafarm/commit/60bb60340ea2723e77d0ff7883594a8f52523945))
* **designer:** clean up markdown responses and typing annimation ([cce0ea4](https://github.com/llama-farm/llamafarm/commit/cce0ea450299cc3d0941e0f21dbd8cc59b875517))
* **designer:** designer project name validation ([#385](https://github.com/llama-farm/llamafarm/issues/385)) ([96c5499](https://github.com/llama-farm/llamafarm/commit/96c5499f403b4be2da1f1057d1d5d5d2c985b09c))
* **designer:** empty state to prompts tables/page and styling ([df9a732](https://github.com/llama-farm/llamafarm/commit/df9a73290f7d4a3f6f0f0cf668552ef0d6d685e5))
* **designer:** fix build error ([5d3c6a9](https://github.com/llama-farm/llamafarm/commit/5d3c6a9f08ddcc20df85da184a524f0c7d072cc1))
* **designer:** hide package button for now ([6e344f8](https://github.com/llama-farm/llamafarm/commit/6e344f83dd98559b77cc7a7ee4abe92dc4806216))
* **designer:** hide versions card and versions page ([3c88d7a](https://github.com/llama-farm/llamafarm/commit/3c88d7a8a6a84b2fb963e236fecabbbb807ed350))
* **designer:** hook up prompt sets to new config format ([bc28460](https://github.com/llama-farm/llamafarm/commit/bc28460e62c8c7f1d7a49c73214a7db1893ab527))
* **designer:** loading state for project switching and shorter auto-response ([7871ac0](https://github.com/llama-farm/llamafarm/commit/7871ac0fcf719f29b72d0f3eb928ccdc7ce62ab3))
* **designer:** persist default model and robust config updates\n\n- Persist default model via updateProject\n- Handle deleting default model and clear runtime.default_model\n- Add optimistic rollback for add/toggle/clear/delete UI state\n- Harden config-&gt;InferenceModel mapping with fallbacks\n- Replace Docker-only base_url with localhost for local models\n\nfix(dashboard): guard default model display without any-cast\n\nchore(a11y): tie labels to inputs in model dialog ([cdf2eeb](https://github.com/llama-farm/llamafarm/commit/cdf2eebc9abcfe0944d35a28c31d6c19a9fb7b54))
* **designer:** polish markdown render to have less spacing in responses ([72ec926](https://github.com/llama-farm/llamafarm/commit/72ec926952908ed8b63838ce1ff57a5ec2572032))
* **designer:** Project information in designer ([0bd89ba](https://github.com/llama-farm/llamafarm/commit/0bd89ba03faa4e779f7e8bbe4578a98b73d13916))
* **designer:** re-hook models count to the dashboard count + write function for model prompt sets ([9d7daa1](https://github.com/llama-farm/llamafarm/commit/9d7daa1e45294633c40f42b138141754552d9e2f))
* **designer:** remove @codemirror/lint from vite config - package not installed ([57fa504](https://github.com/llama-farm/llamafarm/commit/57fa504020b0e4648aaf3d269b8d347224282337))
* **designer:** remove ajv and ajv-formats from vite config - packages not installed or used ([0064070](https://github.com/llama-farm/llamafarm/commit/006407034ae3ee3d36eb561970465830ffa6b28d))
* **designer:** remove prompt IDs ([790c632](https://github.com/llama-farm/llamafarm/commit/790c632adf0e4c1b4d0f35cb0f2434a561c6edcf))
* **designer:** remove session ids ([bf048d7](https://github.com/llama-farm/llamafarm/commit/bf048d71830155df4069b9c60164599d624b723a))
* **designer:** resolve TypeScript build errors in Models.tsx ([252f3e0](https://github.com/llama-farm/llamafarm/commit/252f3e0994d9d5001ad3af25b2196e969a97ec29)), closes [#372](https://github.com/llama-farm/llamafarm/issues/372)
* **designer:** revert destructive color for accessibility compliance ([f4f7cb9](https://github.com/llama-farm/llamafarm/commit/f4f7cb9a07246bf25b55ae8c555bb164f5c99fb8))
* **designer:** slim down this PR (maybe) ([2a9bc60](https://github.com/llama-farm/llamafarm/commit/2a9bc6008f434b0b0cca57506a0bca48f5f589e0))
* **designer:** sourcery comments and quodo comments and remove session stuff ([262215a](https://github.com/llama-farm/llamafarm/commit/262215a0911779277041d2e2fd70309bcdc2aaed))
* **designer:** update sample projects to match new config standard ([#373](https://github.com/llama-farm/llamafarm/issues/373)) ([3d99e03](https://github.com/llama-farm/llamafarm/commit/3d99e03437435a06453569f092d640cff66c272a))
* **docs:** designer updates + FDA example ([#386](https://github.com/llama-farm/llamafarm/issues/386)) ([3842ae1](https://github.com/llama-farm/llamafarm/commit/3842ae1ccbfa11afc84a6da68025f6a20fa8618c))
* **e2e:** add runtime field to CLI config struct ([a648d2a](https://github.com/llama-farm/llamafarm/commit/a648d2ada24b92de0eda0046b91b314f115cf9c7)), closes [#372](https://github.com/llama-farm/llamafarm/issues/372)
* fixed test and build errors ([8b44134](https://github.com/llama-farm/llamafarm/commit/8b4413474ba56455f093f57d099011cfcb5e3761))
* OllamaEmbedder respects configured dimension from llamafarm.yaml ([#338](https://github.com/llama-farm/llamafarm/issues/338)) ([ab15fc8](https://github.com/llama-farm/llamafarm/commit/ab15fc8a739b21c5da7bfeed1594ee2552f3a453))
* RAG worker Ollama connection in Docker environments ([#347](https://github.com/llama-farm/llamafarm/issues/347)) ([043c55e](https://github.com/llama-farm/llamafarm/commit/043c55e356b8b0802bd7c93c487791bd709e911d))
* **server:** remove localhost address in seed project ([8b0ab07](https://github.com/llama-farm/llamafarm/commit/8b0ab0772fd1477139b3d0be415488eab9b9e7ae))
* url rewriting logic ([#342](https://github.com/llama-farm/llamafarm/issues/342)) ([c1a2721](https://github.com/llama-farm/llamafarm/commit/c1a272106056462e97ef32b79340d758d1ec23d8))
* use typed structs for config serialization in CLI init command ([#384](https://github.com/llama-farm/llamafarm/issues/384)) ([529f76a](https://github.com/llama-farm/llamafarm/commit/529f76a9b4008e31ace40de7adf1cccac667bab9))


### Miscellaneous Chores

* release 0.0.10 ([5981f3b](https://github.com/llama-farm/llamafarm/commit/5981f3b54c63a461187813e7dea39c513f77bfa6))

## [0.0.9](https://github.com/llama-farm/llamafarm/compare/v0.0.8...v0.0.9) (2025-10-20)


### Features

* **cli:** Cli TUI redesign, multi-database coverage in CLI, API ([#312](https://github.com/llama-farm/llamafarm/issues/312)) ([ed59d9e](https://github.com/llama-farm/llamafarm/commit/ed59d9e437d666792b7f3566bec90f71d313c604))
* **cli:** make cli resonsive, reorganzie commands add a whole bunch of stuff ([035c9b7](https://github.com/llama-farm/llamafarm/commit/035c9b76d05b95f242f19d00ce1d35cb16752d60))
* **designer:** hook up db to designer ([aa84d44](https://github.com/llama-farm/llamafarm/commit/aa84d444bb2b6694942bee62afeed42ee8a79898))
* **designer:** intro databases - replace rag tab with databases pt 2 ([2df55ed](https://github.com/llama-farm/llamafarm/commit/2df55edde29e09ec4c9d2b5514bb5c90390fde62))
* **designer:** move processing strat to data tab pt 1 ([585c087](https://github.com/llama-farm/llamafarm/commit/585c0877e477a63ce3ef030f1a6b14e0f6e681ab))
* Implement named prompt sets with per-model selection ([#324](https://github.com/llama-farm/llamafarm/issues/324)) ([5f59f03](https://github.com/llama-farm/llamafarm/commit/5f59f03108cdfeeb62dcdad85aef1b40f65b4404))
* Multi-model examples and Lemonade runtime config ([#310](https://github.com/llama-farm/llamafarm/issues/310)) ([ead2453](https://github.com/llama-farm/llamafarm/commit/ead245379274aaa46f4e13254543119b2d7d9c72))


### Bug Fixes

* **build:** .dockerignore file matching ([#316](https://github.com/llama-farm/llamafarm/issues/316)) ([4c26b27](https://github.com/llama-farm/llamafarm/commit/4c26b271be3f010d7288b2c4ad6ad875cfd1d895))
* **cli:** address scourcery comments, cubic, quodo, build issues ([33ba216](https://github.com/llama-farm/llamafarm/commit/33ba216269c18db50087598ee412e55a116e5b3a))
* **cli:** attempt code cleanup ([1920a02](https://github.com/llama-farm/llamafarm/commit/1920a024760ee11dd90d2f65f2a89860a0a55a38))
* **cli:** make health check real not hardcoded ([413245e](https://github.com/llama-farm/llamafarm/commit/413245e624116ee8167f0d29cba21b9ba3cedf8e))
* **cli:** menu polish commands and ux ([78f8c9c](https://github.com/llama-farm/llamafarm/commit/78f8c9c61ab89ad503606981b681c1e83f7b4a25))
* **cli:** menu update cmnd k portion ([67d1e07](https://github.com/llama-farm/llamafarm/commit/67d1e0715cce09761f8c6897578bed63dbc4e8ec))
* **designer:** cubic sourcery and quodo comments and build errors ([39b98a1](https://github.com/llama-farm/llamafarm/commit/39b98a1387c1a6ec6e82bcb8ea505f709654d00c))
* **designer:** fix build error ([4c600f9](https://github.com/llama-farm/llamafarm/commit/4c600f9d656886d52588986c95d56ed30f0da29f))
* **designer:** new embedding and retrival strats can be added to db ([5e53da5](https://github.com/llama-farm/llamafarm/commit/5e53da50fc0ce67da80bf61acb9c25a028d32343))
* update database configuration ([#313](https://github.com/llama-farm/llamafarm/issues/313)) ([80a980c](https://github.com/llama-farm/llamafarm/commit/80a980c1d7946cc3a096a1e9a4c6d7afef5a79d7))


### Miscellaneous Chores

* release 0.0.8 ([984da25](https://github.com/llama-farm/llamafarm/commit/984da2503de85875e85b70a52be347ac20942263))
* release 0.0.9 ([53f4f1a](https://github.com/llama-farm/llamafarm/commit/53f4f1a60fd4217502ca58d68deaa9275a7549bb))

## [0.0.8](https://github.com/llama-farm/llamafarm/compare/v0.0.8...v0.0.8) (2025-10-20)


### Features

* **cli:** Cli TUI redesign, multi-database coverage in CLI, API ([#312](https://github.com/llama-farm/llamafarm/issues/312)) ([ed59d9e](https://github.com/llama-farm/llamafarm/commit/ed59d9e437d666792b7f3566bec90f71d313c604))
* **cli:** make cli resonsive, reorganzie commands add a whole bunch of stuff ([035c9b7](https://github.com/llama-farm/llamafarm/commit/035c9b76d05b95f242f19d00ce1d35cb16752d60))
* **designer:** hook up db to designer ([aa84d44](https://github.com/llama-farm/llamafarm/commit/aa84d444bb2b6694942bee62afeed42ee8a79898))
* **designer:** intro databases - replace rag tab with databases pt 2 ([2df55ed](https://github.com/llama-farm/llamafarm/commit/2df55edde29e09ec4c9d2b5514bb5c90390fde62))
* **designer:** move processing strat to data tab pt 1 ([585c087](https://github.com/llama-farm/llamafarm/commit/585c0877e477a63ce3ef030f1a6b14e0f6e681ab))
* Implement named prompt sets with per-model selection ([#324](https://github.com/llama-farm/llamafarm/issues/324)) ([5f59f03](https://github.com/llama-farm/llamafarm/commit/5f59f03108cdfeeb62dcdad85aef1b40f65b4404))
* Multi-model examples and Lemonade runtime config ([#310](https://github.com/llama-farm/llamafarm/issues/310)) ([ead2453](https://github.com/llama-farm/llamafarm/commit/ead245379274aaa46f4e13254543119b2d7d9c72))


### Bug Fixes

* **build:** .dockerignore file matching ([#316](https://github.com/llama-farm/llamafarm/issues/316)) ([4c26b27](https://github.com/llama-farm/llamafarm/commit/4c26b271be3f010d7288b2c4ad6ad875cfd1d895))
* **cli:** address scourcery comments, cubic, quodo, build issues ([33ba216](https://github.com/llama-farm/llamafarm/commit/33ba216269c18db50087598ee412e55a116e5b3a))
* **cli:** attempt code cleanup ([1920a02](https://github.com/llama-farm/llamafarm/commit/1920a024760ee11dd90d2f65f2a89860a0a55a38))
* **cli:** make health check real not hardcoded ([413245e](https://github.com/llama-farm/llamafarm/commit/413245e624116ee8167f0d29cba21b9ba3cedf8e))
* **cli:** menu polish commands and ux ([78f8c9c](https://github.com/llama-farm/llamafarm/commit/78f8c9c61ab89ad503606981b681c1e83f7b4a25))
* **cli:** menu update cmnd k portion ([67d1e07](https://github.com/llama-farm/llamafarm/commit/67d1e0715cce09761f8c6897578bed63dbc4e8ec))
* **designer:** cubic sourcery and quodo comments and build errors ([39b98a1](https://github.com/llama-farm/llamafarm/commit/39b98a1387c1a6ec6e82bcb8ea505f709654d00c))
* **designer:** fix build error ([4c600f9](https://github.com/llama-farm/llamafarm/commit/4c600f9d656886d52588986c95d56ed30f0da29f))
* **designer:** new embedding and retrival strats can be added to db ([5e53da5](https://github.com/llama-farm/llamafarm/commit/5e53da50fc0ce67da80bf61acb9c25a028d32343))
* update database configuration ([#313](https://github.com/llama-farm/llamafarm/issues/313)) ([80a980c](https://github.com/llama-farm/llamafarm/commit/80a980c1d7946cc3a096a1e9a4c6d7afef5a79d7))


### Miscellaneous Chores

* release 0.0.8 ([984da25](https://github.com/llama-farm/llamafarm/commit/984da2503de85875e85b70a52be347ac20942263))

## [0.0.8](https://github.com/llama-farm/llamafarm/compare/v0.0.7...v0.0.8) (2025-10-13)


### Features

* add multi-model runtime support with API and CLI integration ([#263](https://github.com/llama-farm/llamafarm/issues/263)) ([9630ed3](https://github.com/llama-farm/llamafarm/commit/9630ed3f6b76a4a18f2394ce6fdccea682c95480))
* auto generate datamodel ([#278](https://github.com/llama-farm/llamafarm/issues/278)) ([f1d3a10](https://github.com/llama-farm/llamafarm/commit/f1d3a102628d59c1ca56c03e8123ba4bd183381e))
* **designer:** add working one liner for entire project (CLI and designer) upgrade to new version ([6c72a79](https://github.com/llama-farm/llamafarm/commit/6c72a79e6cfad3b57040bfdc2370c98d7bb069c4))
* **designer:** allow prompts to read and write to config, allow edit/add/delete from designer ([97943c7](https://github.com/llama-farm/llamafarm/commit/97943c7270f9db0fe134a4fb120d14abe57ad254))
* **designer:** chat switcher for small screens ([76ca599](https://github.com/llama-farm/llamafarm/commit/76ca599c5969f7b4e18e1dc01f5b869d5ff92755))
* **designer:** connect prompts to pull in project config prompts to prompts page ([befbb19](https://github.com/llama-farm/llamafarm/commit/befbb195aff30d6fdf197d88cc3d247d4b036573))
* **designer:** Fix responsiveness issues ([a83f8a3](https://github.com/llama-farm/llamafarm/commit/a83f8a365a3a608efbde4ebcc77669994d090a95))
* **designer:** Hook up prompts from config ([c3dc4bf](https://github.com/llama-farm/llamafarm/commit/c3dc4bfeac45f3a59de0ac87cfcb2ea9199f791f))
* **designer:** Hook up upgrade api to designer UI ([44457be](https://github.com/llama-farm/llamafarm/commit/44457bee7cc45ce15142c2277ea586416b326b8d))
* **designer:** update current version modal to fir new upgrade exp ([5d020e1](https://github.com/llama-farm/llamafarm/commit/5d020e179881d3b3d6ec11d6851c9d219b3afedf))
* **designer:** update update modal to show success state and view release notes ([a4faa9b](https://github.com/llama-farm/llamafarm/commit/a4faa9b92e8c50ac568a3916cc573403e62cecbc))


### Bug Fixes

* **cli:** windows path for docker ([#300](https://github.com/llama-farm/llamafarm/issues/300)) ([4d35090](https://github.com/llama-farm/llamafarm/commit/4d3509071a5c5d5dafa085991435a9cf25a51c81))
* **designer:** add helper text to prompts modal ([2ac2420](https://github.com/llama-farm/llamafarm/commit/2ac24205f805b8eb905411284617cd328be1f995))
* **designer:** addressing comments, addressing problems fixing comments created, adding bug fix to expandable chat window ([041c58a](https://github.com/llama-farm/llamafarm/commit/041c58a13c9c48346876c58b804dc909f43808eb))
* **designer:** addressing cubic and cursor comments ([90746bd](https://github.com/llama-farm/llamafarm/commit/90746bd70bc650c27020faeb288133f2ce02b5c0))
* **designer:** addressing sourcery comments ([38ca807](https://github.com/llama-farm/llamafarm/commit/38ca807feaf7720f21f5e74e3540eda6850091c9))
* **designer:** addressing sourcery comments ([30a20aa](https://github.com/llama-farm/llamafarm/commit/30a20aaebfa9c13a6cd0fc392ad4cfc74fd7f2a5))
* **designer:** addressing sourecery comments ([9fc3bf9](https://github.com/llama-farm/llamafarm/commit/9fc3bf9b2433b4ccc22ae711e8e1087205ca71de))
* **designer:** button padding final bug I hope ([7bea713](https://github.com/llama-farm/llamafarm/commit/7bea713d628f9267e97714784480093fc602d596))
* **designer:** collapse icon broke again ([14e3a0c](https://github.com/llama-farm/llamafarm/commit/14e3a0c582f73a99e3d820c99f79cc92b8215fa8))
* **designer:** cubic and cursor comments ([d34e804](https://github.com/llama-farm/llamafarm/commit/d34e8044d75588dde3d1012c485c2bfc84094d1d))
* **designer:** dashboard responsivness ([2ffebb3](https://github.com/llama-farm/llamafarm/commit/2ffebb36ec4923303a3bcbe18fb11c5350e09635))
* **designer:** hook up example datasets to import sample dataset flow ([3c8cee0](https://github.com/llama-farm/llamafarm/commit/3c8cee069a5f68c3e51184e4980d6f805dd26f37))
* **designer:** hook up example datasets to import sample dataset flow ([b7b25de](https://github.com/llama-farm/llamafarm/commit/b7b25ded236ecf994cd2ef8285698816800949ab))
* **designer:** nav responsivness ([d74c21a](https://github.com/llama-farm/llamafarm/commit/d74c21aacf22b1aa57e36ee6f8a37ca70d5db526))
* **designer:** one more comment ([0d93ac8](https://github.com/llama-farm/llamafarm/commit/0d93ac86bd07ccc66c6db918d68297cf42c73050))
* **designer:** remove unused variables to resolve TS6133 in Data and ImportSampleDatasetModal ([185b3f9](https://github.com/llama-farm/llamafarm/commit/185b3f97f9ba381b6b6bcaaee585b435aa10eeda))
* **designer:** responsive issues are never ending - fixed edit project modal and sample projects ([71ae6d1](https://github.com/llama-farm/llamafarm/commit/71ae6d183e146f6ac70d59995682da00913711ee))
* **designer:** responsivness on other project pages (prompts, data, models, rag) and modals and breadcrumb pages ... ([bbf4b50](https://github.com/llama-farm/llamafarm/commit/bbf4b50da55e9c93e7d9245dac63da24cf990953))
* **rag:** docx parser fails due to missing parse_blob ([#293](https://github.com/llama-farm/llamafarm/issues/293)) ([2a7c04d](https://github.com/llama-farm/llamafarm/commit/2a7c04dbe265def6ed8b7d2ca6587d404c0b94f9))


### Miscellaneous Chores

* release 0.0.8 ([30d7f89](https://github.com/llama-farm/llamafarm/commit/30d7f89f4e0bfbcd998820a1b699aef954457348))

## [0.0.7](https://github.com/llama-farm/llamafarm/compare/v0.0.6...v0.0.7) (2025-10-07)


### Features

* **designer:** Config editor updates ([#269](https://github.com/llama-farm/llamafarm/issues/269)) ([dfb2282](https://github.com/llama-farm/llamafarm/commit/dfb2282548e78953406e7aad26195110e86ebb31))
* **designer:** re-do import example projects or datasets + empty states and data import ([8a2f50e](https://github.com/llama-farm/llamafarm/commit/8a2f50ec6bd97135ef242348a8b9c8f7ae7112a4))
* **examples:** add manifest.yaml for quick, many-small-file, large-complex, and mixed-format samples ([4d82997](https://github.com/llama-farm/llamafarm/commit/4d8299793ed6594a025ef3aeccfb41ee9da7a0e1))


### Bug Fixes

* **cli:** don't show seed status errors in project chat mode ([#286](https://github.com/llama-farm/llamafarm/issues/286)) ([a6581f1](https://github.com/llama-farm/llamafarm/commit/a6581f1440161faf980af2595640f09dbe97830f))
* **cli:** improve rag startup orchestration ([#275](https://github.com/llama-farm/llamafarm/issues/275)) ([469aabe](https://github.com/llama-farm/llamafarm/commit/469aabe3a64af99868c0f4a2f80787650d8b46b4))
* **cli:** incorrect version string ([#272](https://github.com/llama-farm/llamafarm/issues/272)) ([35a268b](https://github.com/llama-farm/llamafarm/commit/35a268b481904dc4195a3753dcf35726f4707f5c))
* **designer:** build errors and cubic comments ([1fb40d0](https://github.com/llama-farm/llamafarm/commit/1fb40d00f9287e34dc10f101c7a37076b64b0dd3))
* **designer:** build errors fix ([8f9bf87](https://github.com/llama-farm/llamafarm/commit/8f9bf87077639c452da34e581e5c02a5a718f781))
* **designer:** update missing descriptions and meta data for examples ([f248a3a](https://github.com/llama-farm/llamafarm/commit/f248a3ad65588a27400298acc4f1693042c10d98))
* **merge:** resolve conflicts across schema, templates, examples, and server routers; align with new RAG schema and upgrades API ([985134d](https://github.com/llama-farm/llamafarm/commit/985134d4d1818b4ec7bfb03b992bb706f6b51bbd))
* PDF and Markdown parsers now properly chunk documents ([#276](https://github.com/llama-farm/llamafarm/issues/276)) ([c1fb5bc](https://github.com/llama-farm/llamafarm/commit/c1fb5bca2dff3e0b1036617ca3924616c0a7b4ee))
* **rag:** startup ([#273](https://github.com/llama-farm/llamafarm/issues/273)) ([f79d7d0](https://github.com/llama-farm/llamafarm/commit/f79d7d0fbd5a1bc021d304382c16f12f49cbf3fe))


### Miscellaneous Chores

* release 0.0.7 ([19cacf2](https://github.com/llama-farm/llamafarm/commit/19cacf2d479066439ae897e07e2ecd73dab222f5))

## [0.0.6](https://github.com/llama-farm/llamafarm/compare/v0.0.5...v0.0.6) (2025-10-02)


### Features

* **cli:** add curl preview and processing heartbeat ([#243](https://github.com/llama-farm/llamafarm/issues/243)) ([f4d1c59](https://github.com/llama-farm/llamafarm/commit/f4d1c5921f63db06c4ff8c9b0e2e3cd3ce4c71c7))
* **cli:** add upgrade awareness ([#248](https://github.com/llama-farm/llamafarm/issues/248)) ([6955a8f](https://github.com/llama-farm/llamafarm/commit/6955a8f2d27ab925f58dddddf812ab938848b71e))
* **designer:** designer onboarding questions and editable project details ([e778400](https://github.com/llama-farm/llamafarm/commit/e77840029cd6dea62a4e6978b15f27946a8be29b))
* **designer:** project details and editor on dashboard + dashboard clean up ([68523d0](https://github.com/llama-farm/llamafarm/commit/68523d0e0ec5ea80323eec8c3e7123fb50c430c0))
* **designer:** update create project flow, edit modal to work, proejct create loading state, ([2931226](https://github.com/llama-farm/llamafarm/commit/2931226bf45f719cc5c9ecf519d0439d05769494))
* Enhanced CLI chat experience with dev mode, greetings, and RAG fixes ([#254](https://github.com/llama-farm/llamafarm/issues/254)) ([7ad8f78](https://github.com/llama-farm/llamafarm/commit/7ad8f78c49cdbbeb9db9a45408091e4a6d495d21))


### Bug Fixes

* **designer:** addressing sourcery comments ([32211de](https://github.com/llama-farm/llamafarm/commit/32211debbb045c86bb2cb2e2098427bd9cf9f30b))
* **designer:** build error - remove unused svg ([7eadb74](https://github.com/llama-farm/llamafarm/commit/7eadb748abe91d36d3162a56cd021a92c703a614))
* **designer:** persist project details ([5888ff4](https://github.com/llama-farm/llamafarm/commit/5888ff4b1852a3948ff216765b68e5bac4fba8a2))
* **designer:** remove description to fix build error ([075af8f](https://github.com/llama-farm/llamafarm/commit/075af8f7088f3c1ffeaf52f527216283f2704a1a))
* health check ([#261](https://github.com/llama-farm/llamafarm/issues/261)) ([1586fc3](https://github.com/llama-farm/llamafarm/commit/1586fc31598b3e76a8ea3f62ce43f10f74fea4d9))
* **rag:** unable to process data ([#257](https://github.com/llama-farm/llamafarm/issues/257)) ([9fea950](https://github.com/llama-farm/llamafarm/commit/9fea950d6fdce6309d6ea8187399516ca1c7cfe6))
* remove nulls and invalid values from llamafarm.yaml ([#264](https://github.com/llama-farm/llamafarm/issues/264)) ([a98ef40](https://github.com/llama-farm/llamafarm/commit/a98ef40173d2f53941c6e549756360f057048eac))
* resolve designer chat echo detection and unhelpful responses ([#256](https://github.com/llama-farm/llamafarm/issues/256)) ([284bc34](https://github.com/llama-farm/llamafarm/commit/284bc34a24f8952eb3db38584f948f630617136e))
* show status indicator for failures ([#266](https://github.com/llama-farm/llamafarm/issues/266)) ([fc72d7d](https://github.com/llama-farm/llamafarm/commit/fc72d7d866904f66474c4be61beb73bbc5b238a6))


### Miscellaneous Chores

* release 0.0.6 ([a2fd333](https://github.com/llama-farm/llamafarm/commit/a2fd3332df86efdaddf633cf466ca70686724d93))

## [0.0.5](https://github.com/llama-farm/llamafarm/compare/v0.0.4...v0.0.5) (2025-09-30)


### Features

* add directory upload capability to CLI datasets ingest command ([#223](https://github.com/llama-farm/llamafarm/issues/223)) ([5c54648](https://github.com/llama-farm/llamafarm/commit/5c54648e6d5a6ffc2ec7d031ec9265cbf91ce391))
* add flexible agent handlers for rag support ([#233](https://github.com/llama-farm/llamafarm/issues/233)) ([127710b](https://github.com/llama-farm/llamafarm/commit/127710b087d394f2a748aa07da2004dca02029f9))
* **cli:** uniform commands ([#215](https://github.com/llama-farm/llamafarm/issues/215)) ([337391a](https://github.com/llama-farm/llamafarm/commit/337391afd28adbc9d50907b69aac33818bd53a77))
* **cli:** update the cli command structure ([#229](https://github.com/llama-farm/llamafarm/issues/229)) ([94746e5](https://github.com/llama-farm/llamafarm/commit/94746e566b440ba24cd3e2f651bcd4284c10b594))
* **designer:** add 'included file types' field to parser and extractor add/edit modals + modal restructure ([781e1bd](https://github.com/llama-farm/llamafarm/commit/781e1bd846d859d689aafdf5528cdf826dcb68e4))
* **designer:** better dummy data ([87fc352](https://github.com/llama-farm/llamafarm/commit/87fc352d8a1267c0eaed27b1e9ed2b78dde3ddb7))
* **designer:** check for updates from home page ([83f0c81](https://github.com/llama-farm/llamafarm/commit/83f0c8114e10d04002980e7d586d448860a2f54a))
* **designer:** client side session storage ([#152](https://github.com/llama-farm/llamafarm/issues/152)) ([23f06c4](https://github.com/llama-farm/llamafarm/commit/23f06c42d9c66e2dd226a41ddc69c24bc6763b88))
* **designer:** import sample datasets (from sample projects) in data tab ([5195b8e](https://github.com/llama-farm/llamafarm/commit/5195b8e72df0719da6aaa1a7b47a43839076940e))
* **designer:** reprocess assigned datasets when any changes are made to processing strategy ([93932da](https://github.com/llama-farm/llamafarm/commit/93932da49ab6e635a03b49f50b65660b02320fc7))
* **designer:** sample projects page ([d8eec60](https://github.com/llama-farm/llamafarm/commit/d8eec60af5a881722a1270f4412701649658c504))
* **designer:** upgrade banner and toast for new versions ([00eb4cd](https://github.com/llama-farm/llamafarm/commit/00eb4cd00e316b7e4de6e797512c433a9c53262f))
* **rag:** run in separate container ([#181](https://github.com/llama-farm/llamafarm/issues/181)) ([8ba6409](https://github.com/llama-farm/llamafarm/commit/8ba6409536473f756f02c99a68b8cfe1931a5d3b))


### Bug Fixes

* address bugs with rag in Docker ([#213](https://github.com/llama-farm/llamafarm/issues/213)) ([f160b79](https://github.com/llama-farm/llamafarm/commit/f160b796cf5838f018b724ecc5ea2b7d30cf3e03))
* **ci:** resolve merge conflicts in docker workflow; align tags and services to main ([458eb08](https://github.com/llama-farm/llamafarm/commit/458eb08b1080c9a7e772ef4a617040a471934b94))
* **cli:** address bugs with interactive chat viewport ([#206](https://github.com/llama-farm/llamafarm/issues/206)) ([de4f46b](https://github.com/llama-farm/llamafarm/commit/de4f46bb4b9d85216c374991daafac4b7a9f055b))
* **cli:** ensure project sync runs for required commands ([#216](https://github.com/llama-farm/llamafarm/issues/216)) ([f27a29b](https://github.com/llama-farm/llamafarm/commit/f27a29b7cef5183d41b50e41ff88db3f420ab50d))
* **cli:** set docker host mapping for linux ([#242](https://github.com/llama-farm/llamafarm/issues/242)) ([8ccbf7b](https://github.com/llama-farm/llamafarm/commit/8ccbf7bbc23252e067b42c963c4c154b354c72e2))
* **cli:** set user directive in container ([#246](https://github.com/llama-farm/llamafarm/issues/246)) ([e152341](https://github.com/llama-farm/llamafarm/commit/e152341f85a1d9cb41aaa1125098de5dcc894d03))
* **cli:** Windows Docker volume mount path conversion ([#222](https://github.com/llama-farm/llamafarm/issues/222)) ([941a46e](https://github.com/llama-farm/llamafarm/commit/941a46e74c833d698167491116459f22c5756397))
* **config:** update default template ([#225](https://github.com/llama-farm/llamafarm/issues/225)) ([b98508d](https://github.com/llama-farm/llamafarm/commit/b98508d0fea4d76185b9c0dbca8839d81fa82821))
* **designer:** add better individual parser and extractor descriptions to ui + better formating to edit/add ([b7914bc](https://github.com/llama-farm/llamafarm/commit/b7914bc9f1ae29e2b344c969624129b271d8fdf1))
* **designer:** addressing sourcery and cubic comments and build errors ([8c20106](https://github.com/llama-farm/llamafarm/commit/8c2010671aa98314da9c4833bc65df120e7b072e))
* **designer:** another build error fix ([49164bd](https://github.com/llama-farm/llamafarm/commit/49164bdc84297b7cd3c4d5cf128b6f60b6e6b2ba))
* **designer:** another fix ([cd72833](https://github.com/llama-farm/llamafarm/commit/cd72833fd923049cac7d6b1bb8a8070b9c07e80b))
* **designer:** build error fix ([ae2ab01](https://github.com/llama-farm/llamafarm/commit/ae2ab0186161abfd2769695c5c93acedb1de862a))
* **designer:** build errors again ([25994b8](https://github.com/llama-farm/llamafarm/commit/25994b8bdfe231b43629085ca1c3da9212ae5f0d))
* **designer:** dataset cards to display correct processing strategy ([4c540ec](https://github.com/llama-farm/llamafarm/commit/4c540ec27ccfe382944af0c3a4a98ba564c34554))
* **designer:** fix build errors and address sourcery comments ([bf22d3c](https://github.com/llama-farm/llamafarm/commit/bf22d3ca66d62c0056fb858e18ed81df1ca3219e))
* **designer:** header tabs responsiveness - shrink to icons on small screen width ([3aec945](https://github.com/llama-farm/llamafarm/commit/3aec945d029c1c7d0c20a5690f972136e908178b))
* **designer:** human readable extractor/parser names / more context ([7bf69d0](https://github.com/llama-farm/llamafarm/commit/7bf69d059e6a7e95105d425e7bab433c882211bc))
* **designer:** light mode logo wasn't showing, increase logo size in header ([bc2584c](https://github.com/llama-farm/llamafarm/commit/bc2584c0313eecf18015ed91798f60bc9ce5a7e4))
* **designer:** priority cannot go below 0 for parsers/extractors ([07d44a5](https://github.com/llama-farm/llamafarm/commit/07d44a560600f7d9789b9363738e33ba2fada04f))
* **designer:** revert dataset strat issues, remove rag strategy ([179cb28](https://github.com/llama-farm/llamafarm/commit/179cb287cbb26bf9abedd2beecd5bc6a1bf18879))
* **designer:** search by sample project tag ([d162c4b](https://github.com/llama-farm/llamafarm/commit/d162c4b31350c4a5fe0954a4099068e067b1784b))
* **designer:** sourcery comments addressed and build error addressed ([1204eb5](https://github.com/llama-farm/llamafarm/commit/1204eb5906682a0c49adfb9f4a3f4d7aa35bea15))
* **docs:** oss license and py version badges ([dae37fc](https://github.com/llama-farm/llamafarm/commit/dae37fc1a61c3bd120ea8ca77eef43b89d9b9037))
* **docs:** remove home page, polish header sizes and alignment ([97194ee](https://github.com/llama-farm/llamafarm/commit/97194ee8ac4322b79a10b3cd0e60448cba0ad21c))
* **docs:** remove home page, polish header sizes and alignment ([10e8cbe](https://github.com/llama-farm/llamafarm/commit/10e8cbe8cfefada2b1f187fa3a8f94f9fa8efc52))
* **docs:** sourcery edits to simplfy styling ([0fdb6d8](https://github.com/llama-farm/llamafarm/commit/0fdb6d867813b691637b9ede906ef47d66365c5c))
* **rag:** store data in proper location within project dir ([#217](https://github.com/llama-farm/llamafarm/issues/217)) ([80d6125](https://github.com/llama-farm/llamafarm/commit/80d61259b4722135ebaefb2fc1e0f825bd5c8950))
* remove rag_strategy references ([#198](https://github.com/llama-farm/llamafarm/issues/198)) ([84465ec](https://github.com/llama-farm/llamafarm/commit/84465ec5f558a4524ac64bdff90ce5e72e2c0451))
* resolve ollama integration issues and instructor mode configuration ([#196](https://github.com/llama-farm/llamafarm/issues/196)) ([04633d3](https://github.com/llama-farm/llamafarm/commit/04633d3e110db475c887de4c2d660b169d1e4380))
* **server:** minor health check improvements ([f27a29b](https://github.com/llama-farm/llamafarm/commit/f27a29b7cef5183d41b50e41ff88db3f420ab50d))


### Miscellaneous Chores

* release 0.0.5 ([7faa7e0](https://github.com/llama-farm/llamafarm/commit/7faa7e0b9ed8e08bc8269516164bc0cc653ac54d))

## [0.0.4](https://github.com/llama-farm/llamafarm/compare/v0.0.3...v0.0.4) (2025-09-22)


### Features

* **cli:** run designer in background on `lf dev` ([#145](https://github.com/llama-farm/llamafarm/issues/145)) ([f868502](https://github.com/llama-farm/llamafarm/commit/f8685023c44e5317085433a60e7824d6e8d46f89))
* **designer:** add project chat to test tab ([9a3944d](https://github.com/llama-farm/llamafarm/commit/9a3944deae2eac892031c9f963b4eab875d5ed6d))
* **designer:** add/edit parsers and extractors with correct fields from the rag schema.yaml ([0aeab41](https://github.com/llama-farm/llamafarm/commit/0aeab4116c034ac4aafa1ab72f35d85d57da0245))
* **designer:** allow muletiple retrival strategies (edit/make default, add flows) ([d33eec0](https://github.com/llama-farm/llamafarm/commit/d33eec0f32f49dec51c2b4447c21fadde3406b74))
* **designer:** allow users to diagnose bad responses or tests via designer chat ([1a35583](https://github.com/llama-farm/llamafarm/commit/1a355836915e2a5f48a7159fdd1c3a21084880bb))
* **designer:** embedding model / strategy show port and more details. update RAG page layout ([568f568](https://github.com/llama-farm/llamafarm/commit/568f56879a541f4307d8430c1f6184266a793dee))
* **designer:** embedding models - edit, change, add - better reflect the config and schema options ([c65295a](https://github.com/llama-farm/llamafarm/commit/c65295a6af95226d93bdce92eaf890b8fd5e7754))
* **designer:** enable chat streaming ([#180](https://github.com/llama-farm/llamafarm/issues/180)) ([93d8c55](https://github.com/llama-farm/llamafarm/commit/93d8c551fd7910bcba54b7f0068d951585750a7c))
* **designer:** generation settings bar ([4112233](https://github.com/llama-farm/llamafarm/commit/4112233672035e6843e0591f064c663f5294133a))
* **designer:** more response detail (prompts sent, thinking steps, temperature and settings, raw chunk data) ([5e74fd3](https://github.com/llama-farm/llamafarm/commit/5e74fd3b8a197dca4abbf928b473312e97aafc1a))
* **designer:** processing strategy - edit add remove parsers and extractors ([2c35c26](https://github.com/llama-farm/llamafarm/commit/2c35c268dd775b32658a3c1f5b37b35ce77b7f21))
* **designer:** Processing strategy view with extractors and parsers tables ([42ea217](https://github.com/llama-farm/llamafarm/commit/42ea217c38b877336b47f2b234f34bc5861b0275))
* **designer:** RAG home page restructure ([4608449](https://github.com/llama-farm/llamafarm/commit/46084495e1fe2ec81cdd056a1149a7940e095e27))
* **designer:** RAG home page restructure ([24b27d2](https://github.com/llama-farm/llamafarm/commit/24b27d2ea96897f07576c0d80543457b8af2b5b0))
* **designer:** reorder header tabs to flow nicer ([31e4da1](https://github.com/llama-farm/llamafarm/commit/31e4da1a5624580c1dd7e016d8ef38ae2fd826fd))
* **designer:** retrieval methods add templates and relevant settings for each from schema ([f22a27a](https://github.com/llama-farm/llamafarm/commit/f22a27a502deebb591358f179a5eef1a97a380fc))
* **designer:** run tests in project chat ([a2b8fc8](https://github.com/llama-farm/llamafarm/commit/a2b8fc8e38ba31e48823597bc1a64edf0e287df6))
* **designer:** update dataset view to include processing strategy instead of RAG strategy ([ce05d32](https://github.com/llama-farm/llamafarm/commit/ce05d3274f46ab24c7270dd22f8ee6ea00d88a89))
* **designer:** updates to 'diagnose' feature ([a3f2d57](https://github.com/llama-farm/llamafarm/commit/a3f2d572943e79b558bdd92620d2f2312c973018))
* Include file names in dataset API responses ([#153](https://github.com/llama-farm/llamafarm/issues/153)) ([d380457](https://github.com/llama-farm/llamafarm/commit/d380457fd2da480393a0a4fdf1f622039b1a604b))
* **parsers:** add comprehensive LlamaIndex parser support ([#126](https://github.com/llama-farm/llamafarm/issues/126)) ([e9ef35b](https://github.com/llama-farm/llamafarm/commit/e9ef35bbfa0a35b02612e41d0ba332fd40e339b2))
* **rag:** enhanced processing feedback and duplicate detection ([#168](https://github.com/llama-farm/llamafarm/issues/168)) ([664ce43](https://github.com/llama-farm/llamafarm/commit/664ce43f4b37c4222f9c75356847c8d8cb82937c))
* **rag:** new strategy approach + lf cli integration  ([#148](https://github.com/llama-farm/llamafarm/issues/148)) ([e645479](https://github.com/llama-farm/llamafarm/commit/e6454791d332241af7a82322e47ea6182ccaaac8))


### Bug Fixes

* add default system prompt to new project templates ([#155](https://github.com/llama-farm/llamafarm/issues/155)) ([275c0ee](https://github.com/llama-farm/llamafarm/commit/275c0eef6084b3a8b3462bbfc7447066958c1af4))
* **deisgner:** addressing co-pilot and sourcery comments + build errors ([d26736d](https://github.com/llama-farm/llamafarm/commit/d26736dff537c9e69d4c725f70dc1cbed60f0373))
* **designer:** address sorcery issues for create project modal changes ([60fa35d](https://github.com/llama-farm/llamafarm/commit/60fa35d8bf080cad2f6885b7a0d972774b4cd057))
* **designer:** addressing cubic issues ([229d4af](https://github.com/llama-farm/llamafarm/commit/229d4af4c823fe4f9820418df2d86f2fb20b565c))
* **designer:** addressing racheal comment ([8a434d6](https://github.com/llama-farm/llamafarm/commit/8a434d67474d22810c712e431f6315d90c254279))
* **designer:** build errors ([5ee53fd](https://github.com/llama-farm/llamafarm/commit/5ee53fd61caf727366b8083a627d0860a7c63fb9))
* **designer:** clean up Prompts page to avoid repetition in new test ([4e531f0](https://github.com/llama-farm/llamafarm/commit/4e531f0915a32824f4f3f46d207153e4bf583a9d))
* **designer:** clean up Prompts page to avoid repetition in new test exp and polish ([136a5f2](https://github.com/llama-farm/llamafarm/commit/136a5f2dd8dca2420520302ed8ce366b962b3059))
* **designer:** creating project experience was not working ([d7fd8af](https://github.com/llama-farm/llamafarm/commit/d7fd8af8307190930896726aa81122a0fd79f5d9))
* **designer:** creating project experience was not working ([635badd](https://github.com/llama-farm/llamafarm/commit/635badd5f4935b4015ea3d2244d5f9df3649e2b8))
* **designer:** fix build error ([9376223](https://github.com/llama-farm/llamafarm/commit/93762233e48f9bf606e560bfc9a699315f1804f6))
* **designer:** fix build errors pt 2 ([6d2b0ee](https://github.com/llama-farm/llamafarm/commit/6d2b0eef0a060030cb9dd8365a717d534f870ca3))
* **designer:** hybrid retrieval strategies need to have all the same settings within sub strategies ([5f805af](https://github.com/llama-farm/llamafarm/commit/5f805afcdad67b41080ad19c668386b6bfc3cacd))
* **designer:** parsers and extractors priority numbers were flipped - now 1 is top instead of 100 ([40e3904](https://github.com/llama-farm/llamafarm/commit/40e3904e0090f48b279ddc5bde9a88f6d50067b9))
* **designer:** polishing test experience - empty state and helper texts ([89b3c21](https://github.com/llama-farm/llamafarm/commit/89b3c21af72de23bb80140c6505d38deac3efdfd))
* **designer:** remove redundant line in docker file ([77fa523](https://github.com/llama-farm/llamafarm/commit/77fa5232c37ed576f3839b2278c8d643ebd0ebd6))
* **designer:** swap processing strat and embedding/retrival on rag page ([a0a9ccd](https://github.com/llama-farm/llamafarm/commit/a0a9ccd249a312cdf362502e94263cb448087c42))
* lf run project path + jsonschema validation error output ([#170](https://github.com/llama-farm/llamafarm/issues/170)) ([8880a93](https://github.com/llama-farm/llamafarm/commit/8880a934f909525a40fe895e1a076a39b6646a57))
* **rag:** context accumulation in chat history ([#162](https://github.com/llama-farm/llamafarm/issues/162)) ([870b3c2](https://github.com/llama-farm/llamafarm/commit/870b3c22f90ba384217429bd491d57bff8a81638))


### Miscellaneous Chores

* release 0.0.4 ([1015c6e](https://github.com/llama-farm/llamafarm/commit/1015c6eb1538a4e11906a29ad9a3be30cdec9d5d))

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
