# 🗞️ Changelog

## [v1.3.0](https://github.com/python-adaptive/adaptive/tree/v1.3.0)  (2024-04-10)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v1.2.0...v1.3.0)

**Merged pull requests:**

- Replace deprecated numpy aliases [\#458](https://github.com/python-adaptive/adaptive/pull/458) ([eendebakpt](https://github.com/eendebakpt))
- Remove `SKOptLearner` because `scikit-optimize` is unmaintained [\#404](https://github.com/python-adaptive/adaptive/pull/404) ([basnijholt](https://github.com/basnijholt))

## [v1.2.0](https://github.com/python-adaptive/adaptive/tree/v1.2.0) (2024-04-10)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v1.1.0...v1.2.0)

**Closed issues:**

-  Issues with Multiprocess and AsyncRunner in adaptive for Phase Diagram Illustration [\#449](https://github.com/python-adaptive/adaptive/issues/449)
- Create API for just signle process \(No pickle\) [\#442](https://github.com/python-adaptive/adaptive/issues/442)
- Handling with regions unreachable inside the `ConvexHull` in `LearnerND` [\#438](https://github.com/python-adaptive/adaptive/issues/438)
- Use in script with BlockingRunner: get log and/or feedback on progress [\#436](https://github.com/python-adaptive/adaptive/issues/436)

**Merged pull requests:**

- Update CHANGELOG.md for v1.2.0 [\#454](https://github.com/python-adaptive/adaptive/pull/454) ([basnijholt](https://github.com/basnijholt))
- Test Python 3.12 and fix its installation [\#453](https://github.com/python-adaptive/adaptive/pull/453) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#447](https://github.com/python-adaptive/adaptive/pull/447) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Use ruff-format instead of black [\#446](https://github.com/python-adaptive/adaptive/pull/446) ([basnijholt](https://github.com/basnijholt))
- Bump versions to compatible packages in `docs/environment.yml` [\#445](https://github.com/python-adaptive/adaptive/pull/445) ([basnijholt](https://github.com/basnijholt))
- Add `AsyncRunner.block\_until\_done` [\#444](https://github.com/python-adaptive/adaptive/pull/444) ([basnijholt](https://github.com/basnijholt))
- Add `live\_info\_terminal`, closes \#436 [\#441](https://github.com/python-adaptive/adaptive/pull/441) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#434](https://github.com/python-adaptive/adaptive/pull/434) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Add benchmarks page for Learner1D and Learner2D functions [\#405](https://github.com/python-adaptive/adaptive/pull/405) ([basnijholt](https://github.com/basnijholt))

## [v1.1.0](https://github.com/python-adaptive/adaptive/tree/v1.1.0) (2023-08-14)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v1.0.0...v1.1.0)

**Closed issues:**

- large delay when using start\_periodic\_saving  [\#439](https://github.com/python-adaptive/adaptive/issues/439)
- Target function returns NaN [\#435](https://github.com/python-adaptive/adaptive/issues/435)

**Merged pull requests:**

- Ensure periodic saving fires immediately after runner task is finished [\#440](https://github.com/python-adaptive/adaptive/pull/440) ([jbweston](https://github.com/jbweston))
- Add Learner2D loss function 'thresholded\_loss\_factory' [\#437](https://github.com/python-adaptive/adaptive/pull/437) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#433](https://github.com/python-adaptive/adaptive/pull/433) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#431](https://github.com/python-adaptive/adaptive/pull/431) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Add adaptive.utils.daskify [\#422](https://github.com/python-adaptive/adaptive/pull/422) ([basnijholt](https://github.com/basnijholt))

## [v1.0.0](https://github.com/python-adaptive/adaptive/tree/v1.0.0) (2023-05-15)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.15.1...v1.0.0)

**Closed issues:**

- No module named 'typing\_extensions'" [\#394](https://github.com/python-adaptive/adaptive/issues/394)
- Documentation: use cases of coroutine by Learner and Runner not properly explained [\#360](https://github.com/python-adaptive/adaptive/issues/360)

**Merged pull requests:**

- Update CHANGELOG for v1.0.0 [\#429](https://github.com/python-adaptive/adaptive/pull/429) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#426](https://github.com/python-adaptive/adaptive/pull/426) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Allow storing the full sequence in SequenceLearner.to\_dataframe [\#425](https://github.com/python-adaptive/adaptive/pull/425) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#423](https://github.com/python-adaptive/adaptive/pull/423) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Add github-changelog-generator CI workflow [\#421](https://github.com/python-adaptive/adaptive/pull/421) ([basnijholt](https://github.com/basnijholt))
- Use versioningit to work with pyproject.toml and remove setup.py [\#420](https://github.com/python-adaptive/adaptive/pull/420) ([basnijholt](https://github.com/basnijholt))
- Update Python version requirement to 3.9 in accordance with NEP 29 [\#418](https://github.com/python-adaptive/adaptive/pull/418) ([basnijholt](https://github.com/basnijholt))
- Update GitHub Actions CI [\#416](https://github.com/python-adaptive/adaptive/pull/416) ([basnijholt](https://github.com/basnijholt))
- Disable typeguard CI pipeline [\#415](https://github.com/python-adaptive/adaptive/pull/415) ([basnijholt](https://github.com/basnijholt))
- Add mypy to pre-commit and fix all current typing issues [\#414](https://github.com/python-adaptive/adaptive/pull/414) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#413](https://github.com/python-adaptive/adaptive/pull/413) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Use codecov/codecov-action instead of removed codecov package [\#412](https://github.com/python-adaptive/adaptive/pull/412) ([basnijholt](https://github.com/basnijholt))
- Cache the loss of a SequenceLearner [\#411](https://github.com/python-adaptive/adaptive/pull/411) ([basnijholt](https://github.com/basnijholt))
- Only really import packages when needed [\#410](https://github.com/python-adaptive/adaptive/pull/410) ([basnijholt](https://github.com/basnijholt))
- Remove \_RequireAttrsABCMeta metaclass and replace with simple check [\#409](https://github.com/python-adaptive/adaptive/pull/409) ([basnijholt](https://github.com/basnijholt))
- Use the build module [\#407](https://github.com/python-adaptive/adaptive/pull/407) ([basnijholt](https://github.com/basnijholt))
- Avoid asyncio.coroutine error in Readthedocs.org builds [\#406](https://github.com/python-adaptive/adaptive/pull/406) ([basnijholt](https://github.com/basnijholt))
- Bump scikit-optimize in environment.yml [\#403](https://github.com/python-adaptive/adaptive/pull/403) ([basnijholt](https://github.com/basnijholt))
- Replace isort, flake8, and pyupgrade by ruff [\#402](https://github.com/python-adaptive/adaptive/pull/402) ([basnijholt](https://github.com/basnijholt))
- Move to pyproject.toml based install [\#401](https://github.com/python-adaptive/adaptive/pull/401) ([basnijholt](https://github.com/basnijholt))
- Rewrite parts of README, reorder sections, and add features section [\#400](https://github.com/python-adaptive/adaptive/pull/400) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#398](https://github.com/python-adaptive/adaptive/pull/398) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#397](https://github.com/python-adaptive/adaptive/pull/397) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Use nb\_execution\_raise\_on\_error Sphinx myst-nb option [\#396](https://github.com/python-adaptive/adaptive/pull/396) ([basnijholt](https://github.com/basnijholt))
- \[pre-commit.ci\] pre-commit autoupdate [\#395](https://github.com/python-adaptive/adaptive/pull/395) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#393](https://github.com/python-adaptive/adaptive/pull/393) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- \[pre-commit.ci\] pre-commit autoupdate [\#392](https://github.com/python-adaptive/adaptive/pull/392) ([pre-commit-ci[bot]](https://github.com/apps/pre-commit-ci))
- Add nbQA for notebook and docs linting [\#361](https://github.com/python-adaptive/adaptive/pull/361) ([basnijholt](https://github.com/basnijholt))
- Fix HoloViews opts deprecation warnings [\#357](https://github.com/python-adaptive/adaptive/pull/357) ([basnijholt](https://github.com/basnijholt))

## [v0.15.1](https://github.com/python-adaptive/adaptive/tree/v0.15.1) (2022-12-02)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.15.0...v0.15.1)

## [v0.15.0](https://github.com/python-adaptive/adaptive/tree/v0.15.0) (2022-12-02)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.14.2...v0.15.0)

**Closed issues:**

- Fix branch name \(master --\> main\) in binder link in readme [\#383](https://github.com/python-adaptive/adaptive/issues/383)

**Merged pull requests:**

- Rename master -\> main [\#384](https://github.com/python-adaptive/adaptive/pull/384) ([basnijholt](https://github.com/basnijholt))
- Add loss\_goal, npoints\_goal, and an auto\_goal function and use it in the runners [\#382](https://github.com/python-adaptive/adaptive/pull/382) ([basnijholt](https://github.com/basnijholt))
- Add docs section about executing coroutines  [\#364](https://github.com/python-adaptive/adaptive/pull/364) ([juandaanieel](https://github.com/juandaanieel))

## [v0.14.2](https://github.com/python-adaptive/adaptive/tree/v0.14.2) (2022-10-14)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.14.1...v0.14.2)

**Closed issues:**

- Rename master branch to main [\#368](https://github.com/python-adaptive/adaptive/issues/368)

**Merged pull requests:**

- Avoid uncecessary iteration in SequenceLearner [\#380](https://github.com/python-adaptive/adaptive/pull/380) ([jbweston](https://github.com/jbweston))
- Use numbers module from stdlib as type [\#379](https://github.com/python-adaptive/adaptive/pull/379) ([basnijholt](https://github.com/basnijholt))
- Add type-hints to tests and misc [\#378](https://github.com/python-adaptive/adaptive/pull/378) ([basnijholt](https://github.com/basnijholt))
- Add type-hints to Learner2D [\#375](https://github.com/python-adaptive/adaptive/pull/375) ([basnijholt](https://github.com/basnijholt))
- Add type-hints to DataSaver [\#373](https://github.com/python-adaptive/adaptive/pull/373) ([basnijholt](https://github.com/basnijholt))
- Type hint IntegratorLearner [\#372](https://github.com/python-adaptive/adaptive/pull/372) ([basnijholt](https://github.com/basnijholt))
- Add type-hints to BalancingLearner [\#371](https://github.com/python-adaptive/adaptive/pull/371) ([basnijholt](https://github.com/basnijholt))
- Optionally run tests with pandas [\#369](https://github.com/python-adaptive/adaptive/pull/369) ([basnijholt](https://github.com/basnijholt))
- Typehint SequenceLearner [\#366](https://github.com/python-adaptive/adaptive/pull/366) ([basnijholt](https://github.com/basnijholt))

## [v0.14.1](https://github.com/python-adaptive/adaptive/tree/v0.14.1) (2022-10-11)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.14.0...v0.14.1)

**Merged pull requests:**

- Use typing.TypeAlias [\#367](https://github.com/python-adaptive/adaptive/pull/367) ([basnijholt](https://github.com/basnijholt))
- Add Learner.new\(\) method that returns an empty copy of the learner [\#365](https://github.com/python-adaptive/adaptive/pull/365) ([basnijholt](https://github.com/basnijholt))

## [v0.14.0](https://github.com/python-adaptive/adaptive/tree/v0.14.0) (2022-10-05)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.13.2...v0.14.0)

**Closed issues:**

- Stop using atomic writes [\#352](https://github.com/python-adaptive/adaptive/issues/352)

**Merged pull requests:**

- Release 0.14 [\#363](https://github.com/python-adaptive/adaptive/pull/363) ([basnijholt](https://github.com/basnijholt))
- Allow to periodically save with any function [\#362](https://github.com/python-adaptive/adaptive/pull/362) ([basnijholt](https://github.com/basnijholt))
- Update pre-commit versions [\#359](https://github.com/python-adaptive/adaptive/pull/359) ([basnijholt](https://github.com/basnijholt))
- Add getting learner's data as pandas.DataFrame; add learner.to\_dataframe method [\#358](https://github.com/python-adaptive/adaptive/pull/358) ([basnijholt](https://github.com/basnijholt))
- Add transparent logo in WebM format \(for dark mode\) [\#356](https://github.com/python-adaptive/adaptive/pull/356) ([basnijholt](https://github.com/basnijholt))
- Docs in Markdown with Myst and change tutorials to Jupytext notebooks [\#355](https://github.com/python-adaptive/adaptive/pull/355) ([basnijholt](https://github.com/basnijholt))
- Remove scipy deprecation warnings [\#354](https://github.com/python-adaptive/adaptive/pull/354) ([eendebakpt](https://github.com/eendebakpt))
- Replacing atomicwrites with os.write  [\#353](https://github.com/python-adaptive/adaptive/pull/353) ([juandaanieel](https://github.com/juandaanieel))
- Fix class name issue with modern versions of dask.distributed [\#351](https://github.com/python-adaptive/adaptive/pull/351) ([maiani](https://github.com/maiani))

## [v0.13.2](https://github.com/python-adaptive/adaptive/tree/v0.13.2) (2022-05-31)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.13.1...v0.13.2)

**Closed issues:**

- Incompatibility of adaptive \(asyncio\) with python=3.10 [\#344](https://github.com/python-adaptive/adaptive/issues/344)
- Question: plot\_trisurf \(matplotlib\) directly from qhull [\#343](https://github.com/python-adaptive/adaptive/issues/343)
- Allow to choose colormap in learner2D.plot\(\) [\#342](https://github.com/python-adaptive/adaptive/issues/342)
- Warning when importing adaptive [\#321](https://github.com/python-adaptive/adaptive/issues/321)

**Merged pull requests:**

- Run seperate typeguard job \(because it is much slower\) [\#350](https://github.com/python-adaptive/adaptive/pull/350) ([basnijholt](https://github.com/basnijholt))
- Skip ipyparallel test on MacOS [\#349](https://github.com/python-adaptive/adaptive/pull/349) ([basnijholt](https://github.com/basnijholt))
- set loop to None in Python 3.10 [\#348](https://github.com/python-adaptive/adaptive/pull/348) ([basnijholt](https://github.com/basnijholt))
- Switch from Tox to Nox [\#347](https://github.com/python-adaptive/adaptive/pull/347) ([basnijholt](https://github.com/basnijholt))
- use 'from \_\_future\_\_ import annotations' [\#346](https://github.com/python-adaptive/adaptive/pull/346) ([basnijholt](https://github.com/basnijholt))
- Update pre-commit filters versions [\#345](https://github.com/python-adaptive/adaptive/pull/345) ([basnijholt](https://github.com/basnijholt))

## [v0.13.1](https://github.com/python-adaptive/adaptive/tree/v0.13.1) (2021-12-13)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.13.0...v0.13.1)

**Implemented enhancements:**

- Splits up documentations page into "algo+examples" and rest [\#330](https://github.com/python-adaptive/adaptive/pull/330) ([basnijholt](https://github.com/basnijholt))

**Fixed bugs:**

- fix 'asyncio.Task.current\_task' -\> 'asyncio.current\_task' [\#331](https://github.com/python-adaptive/adaptive/pull/331) ([basnijholt](https://github.com/basnijholt))

**Closed issues:**

- Learner1D R -\> R^n - Infinite loop [\#339](https://github.com/python-adaptive/adaptive/issues/339)
- Wrong sign for right boundary with Sympy [\#338](https://github.com/python-adaptive/adaptive/issues/338)

**Merged pull requests:**

- Fix NaN issue for Learner1D R -\> R^n [\#340](https://github.com/python-adaptive/adaptive/pull/340) ([Davide-sd](https://github.com/Davide-sd))
- fix tutorial about using loky.get\_reusable\_executor on Windows [\#336](https://github.com/python-adaptive/adaptive/pull/336) ([basnijholt](https://github.com/basnijholt))
- add a code example to the examples page [\#335](https://github.com/python-adaptive/adaptive/pull/335) ([basnijholt](https://github.com/basnijholt))
- rename usage\_examples -\> gallery [\#334](https://github.com/python-adaptive/adaptive/pull/334) ([basnijholt](https://github.com/basnijholt))
- pin scikit-learn to 0.24.2, because of https://github.com/scikit-optimize/scikit-optimize/issues/1059 [\#333](https://github.com/python-adaptive/adaptive/pull/333) ([basnijholt](https://github.com/basnijholt))
- use jupyter-sphinx instead of custom Sphinx directive [\#332](https://github.com/python-adaptive/adaptive/pull/332) ([basnijholt](https://github.com/basnijholt))
- Add an animated logo that shows the working of Adaptive [\#329](https://github.com/python-adaptive/adaptive/pull/329) ([basnijholt](https://github.com/basnijholt))
- Documentation conda environment update to latest versions [\#328](https://github.com/python-adaptive/adaptive/pull/328) ([basnijholt](https://github.com/basnijholt))
- take out a cut from the 3D sphere, LearnerND example [\#327](https://github.com/python-adaptive/adaptive/pull/327) ([basnijholt](https://github.com/basnijholt))
- Learner1D: return inf loss when the bounds aren't done [\#271](https://github.com/python-adaptive/adaptive/pull/271) ([basnijholt](https://github.com/basnijholt))

## [v0.13.0](https://github.com/python-adaptive/adaptive/tree/v0.13.0) (2021-09-10)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.12.2...v0.13.0)

**Fixed bugs:**

- AverageLearner doesn't work with 0 mean [\#275](https://github.com/python-adaptive/adaptive/issues/275)
- call self.\_process\_futures on canceled futures when BlockingRunner is done [\#320](https://github.com/python-adaptive/adaptive/pull/320) ([basnijholt](https://github.com/basnijholt))
- AverageLearner: fix zero mean [\#276](https://github.com/python-adaptive/adaptive/pull/276) ([basnijholt](https://github.com/basnijholt))

**Closed issues:**

- Runners should tell learner about remaining points at end of run [\#319](https://github.com/python-adaptive/adaptive/issues/319)
- Cryptic error when importing lmfit [\#314](https://github.com/python-adaptive/adaptive/issues/314)
- change CHANGELOG to KeepAChangelog format [\#306](https://github.com/python-adaptive/adaptive/issues/306)
- jupyter notebook kernels dead after running "import adaptive" [\#298](https://github.com/python-adaptive/adaptive/issues/298)
- Emphasis on when to use adaptive in docs [\#297](https://github.com/python-adaptive/adaptive/issues/297)
- GPU acceleration [\#296](https://github.com/python-adaptive/adaptive/issues/296)

**Merged pull requests:**

- Learner1D type hints and add typeguard to pytest tests [\#325](https://github.com/python-adaptive/adaptive/pull/325) ([basnijholt](https://github.com/basnijholt))
- AverageLearner type hints [\#324](https://github.com/python-adaptive/adaptive/pull/324) ([basnijholt](https://github.com/basnijholt))
- Update doc string for resolution\_loss\_function [\#323](https://github.com/python-adaptive/adaptive/pull/323) ([SultanOrazbayev](https://github.com/SultanOrazbayev))
- Update Readme to emphasise when adaptive should be used [\#318](https://github.com/python-adaptive/adaptive/pull/318) ([thomasaarholt](https://github.com/thomasaarholt))
- add to\_numpy methods [\#317](https://github.com/python-adaptive/adaptive/pull/317) ([basnijholt](https://github.com/basnijholt))
- AverageLearner1D added [\#283](https://github.com/python-adaptive/adaptive/pull/283) ([AlvaroGI](https://github.com/AlvaroGI))
- Make LearnerND pickleable [\#272](https://github.com/python-adaptive/adaptive/pull/272) ([basnijholt](https://github.com/basnijholt))
- add a FAQ [\#242](https://github.com/python-adaptive/adaptive/pull/242) ([basnijholt](https://github.com/basnijholt))

## [v0.12.2](https://github.com/python-adaptive/adaptive/tree/v0.12.2) (2021-03-23)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.12.1...v0.12.2)

**Merged pull requests:**

- raise an AttributeError when attribute doesn't exists, closes \#314 [\#315](https://github.com/python-adaptive/adaptive/pull/315) ([basnijholt](https://github.com/basnijholt))

## [v0.12.1](https://github.com/python-adaptive/adaptive/tree/v0.12.1) (2021-03-23)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.12.0...v0.12.1)

**Merged pull requests:**

- write import differently for pypy [\#313](https://github.com/python-adaptive/adaptive/pull/313) ([basnijholt](https://github.com/basnijholt))

## [v0.12.0](https://github.com/python-adaptive/adaptive/tree/v0.12.0) (2021-03-23)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.11.3...v0.12.0)

**Merged pull requests:**

- bump to Python≥3.7 [\#312](https://github.com/python-adaptive/adaptive/pull/312) ([basnijholt](https://github.com/basnijholt))
- lazily evaluate the integrator coefficients [\#311](https://github.com/python-adaptive/adaptive/pull/311) ([basnijholt](https://github.com/basnijholt))
- add resolution\_loss\_function for Learner1D [\#310](https://github.com/python-adaptive/adaptive/pull/310) ([basnijholt](https://github.com/basnijholt))
- add "\(Code\) style fix or documentation update" to .github/pull\_request\_template.md [\#309](https://github.com/python-adaptive/adaptive/pull/309) ([basnijholt](https://github.com/basnijholt))
- remove the requirements from the tutorial landing page [\#308](https://github.com/python-adaptive/adaptive/pull/308) ([basnijholt](https://github.com/basnijholt))
- add change log to the docs [\#307](https://github.com/python-adaptive/adaptive/pull/307) ([basnijholt](https://github.com/basnijholt))
- remove automerge action [\#305](https://github.com/python-adaptive/adaptive/pull/305) ([basnijholt](https://github.com/basnijholt))

## [v0.11.3](https://github.com/python-adaptive/adaptive/tree/v0.11.3) (2021-03-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.11.2...v0.11.3)

**Fixed bugs:**

- ProcessPoolExecutor behaviour on MacOS in interactive environment changed between Python versions [\#301](https://github.com/python-adaptive/adaptive/issues/301)
- can't pickle lru\_cache function with loky [\#292](https://github.com/python-adaptive/adaptive/issues/292)

**Closed issues:**

- Runner fails in the notebook [\#299](https://github.com/python-adaptive/adaptive/issues/299)

**Merged pull requests:**

- add pythonpublish.yml [\#304](https://github.com/python-adaptive/adaptive/pull/304) ([basnijholt](https://github.com/basnijholt))
- fix docs build [\#303](https://github.com/python-adaptive/adaptive/pull/303) ([basnijholt](https://github.com/basnijholt))
- test tox with py39 on Github Actions CI [\#302](https://github.com/python-adaptive/adaptive/pull/302) ([basnijholt](https://github.com/basnijholt))
- make loky a default on Windows and MacOS but not on Linux [\#300](https://github.com/python-adaptive/adaptive/pull/300) ([basnijholt](https://github.com/basnijholt))
- add learner1D.abs\_min\_log\_loss [\#294](https://github.com/python-adaptive/adaptive/pull/294) ([basnijholt](https://github.com/basnijholt))
- bump pre-commit filter dependencies [\#293](https://github.com/python-adaptive/adaptive/pull/293) ([basnijholt](https://github.com/basnijholt))
- fix docs [\#291](https://github.com/python-adaptive/adaptive/pull/291) ([basnijholt](https://github.com/basnijholt))
- update to miniver 0.7.0 [\#290](https://github.com/python-adaptive/adaptive/pull/290) ([basnijholt](https://github.com/basnijholt))
- add `runner.live\_plot\(\)` in README example [\#288](https://github.com/python-adaptive/adaptive/pull/288) ([basnijholt](https://github.com/basnijholt))
- Update pre commit [\#287](https://github.com/python-adaptive/adaptive/pull/287) ([basnijholt](https://github.com/basnijholt))
- Use m2r2 [\#286](https://github.com/python-adaptive/adaptive/pull/286) ([basnijholt](https://github.com/basnijholt))
- temporarily pin scikit-learn\<=0.23.1 [\#285](https://github.com/python-adaptive/adaptive/pull/285) ([basnijholt](https://github.com/basnijholt))
- add .zenodo.json [\#284](https://github.com/python-adaptive/adaptive/pull/284) ([basnijholt](https://github.com/basnijholt))
- always serialize the function using cloudpickle [\#281](https://github.com/python-adaptive/adaptive/pull/281) ([basnijholt](https://github.com/basnijholt))
- add a changelog [\#232](https://github.com/python-adaptive/adaptive/pull/232) ([jbweston](https://github.com/jbweston))

## [v0.11.2](https://github.com/python-adaptive/adaptive/tree/v0.11.2) (2020-08-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.11.1...v0.11.2)

## [v0.11.1](https://github.com/python-adaptive/adaptive/tree/v0.11.1) (2020-08-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.12.0-dev...v0.11.1)

**Closed issues:**

- Release v0.11 [\#277](https://github.com/python-adaptive/adaptive/issues/277)

## [v0.12.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.12.0-dev) (2020-05-20)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.11.0...v0.12.0-dev)

## [v0.11.0](https://github.com/python-adaptive/adaptive/tree/v0.11.0) (2020-05-20)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.11.0-dev...v0.11.0)

**Implemented enhancements:**

- Make Runner work with unhashable points [\#267](https://github.com/python-adaptive/adaptive/issues/267)
- AverageLearner: implement min\_npoints [\#274](https://github.com/python-adaptive/adaptive/pull/274) ([basnijholt](https://github.com/basnijholt))
- make the Runner work with unhashable points [\#268](https://github.com/python-adaptive/adaptive/pull/268) ([basnijholt](https://github.com/basnijholt))
- make learners picklable [\#264](https://github.com/python-adaptive/adaptive/pull/264) ([basnijholt](https://github.com/basnijholt))
- add support for loky [\#263](https://github.com/python-adaptive/adaptive/pull/263) ([basnijholt](https://github.com/basnijholt))
- use \_\_name\_\_ == "\_\_main\_\_" for the MPIPoolExecutor [\#260](https://github.com/python-adaptive/adaptive/pull/260) ([basnijholt](https://github.com/basnijholt))

**Fixed bugs:**

- ipyparallel fails in Python 3.8 [\#249](https://github.com/python-adaptive/adaptive/issues/249)
- Error on windows: daemonic processes are not allowed to have children [\#225](https://github.com/python-adaptive/adaptive/issues/225)
- prevent ImportError due to scikit-optimize and sklearn incompatibility [\#278](https://github.com/python-adaptive/adaptive/pull/278) ([basnijholt](https://github.com/basnijholt))

**Closed issues:**

- add minimum number of points parameter to AverageLearner [\#273](https://github.com/python-adaptive/adaptive/issues/273)
- Release v0.10 [\#258](https://github.com/python-adaptive/adaptive/issues/258)

**Merged pull requests:**

- minimally require ipyparallel 6.2.5 [\#270](https://github.com/python-adaptive/adaptive/pull/270) ([basnijholt](https://github.com/basnijholt))
- fix docs build and pin pyviz\_comms=0.7.2 [\#261](https://github.com/python-adaptive/adaptive/pull/261) ([basnijholt](https://github.com/basnijholt))

## [v0.11.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.11.0-dev) (2020-01-16)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.10.0...v0.11.0-dev)

## [v0.10.0](https://github.com/python-adaptive/adaptive/tree/v0.10.0) (2020-01-15)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.10.0-dev...v0.10.0)

**Implemented enhancements:**

- use tox for testing [\#238](https://github.com/python-adaptive/adaptive/issues/238)
- Time-based stop [\#184](https://github.com/python-adaptive/adaptive/issues/184)
- add a FAQ [\#158](https://github.com/python-adaptive/adaptive/issues/158)

**Fixed bugs:**

- live\_info looks is badly formatted in Jupyterlab [\#250](https://github.com/python-adaptive/adaptive/issues/250)
- SKOptLearner doesn't work for multi variate domain [\#233](https://github.com/python-adaptive/adaptive/issues/233)
- Does not work with lambda functions [\#206](https://github.com/python-adaptive/adaptive/issues/206)

**Merged pull requests:**

- add instructions for installing labextensions for Jupyterlab [\#257](https://github.com/python-adaptive/adaptive/pull/257) ([basnijholt](https://github.com/basnijholt))
- MNT: add vscode config directory to .gitignore [\#255](https://github.com/python-adaptive/adaptive/pull/255) ([tacaswell](https://github.com/tacaswell))
- disable test of runner using distributed [\#253](https://github.com/python-adaptive/adaptive/pull/253) ([basnijholt](https://github.com/basnijholt))
- color the overhead between red and green [\#252](https://github.com/python-adaptive/adaptive/pull/252) ([basnijholt](https://github.com/basnijholt))
- improve the style of the live\_info widget, closes \#250 [\#251](https://github.com/python-adaptive/adaptive/pull/251) ([basnijholt](https://github.com/basnijholt))
- use tox, closes \#238 [\#247](https://github.com/python-adaptive/adaptive/pull/247) ([basnijholt](https://github.com/basnijholt))
- add a Pull Request template [\#246](https://github.com/python-adaptive/adaptive/pull/246) ([basnijholt](https://github.com/basnijholt))
- rename learner.ipynb -\> example-notebook.ipynb [\#241](https://github.com/python-adaptive/adaptive/pull/241) ([basnijholt](https://github.com/basnijholt))
- correct short description in setup.py [\#239](https://github.com/python-adaptive/adaptive/pull/239) ([jbweston](https://github.com/jbweston))
- Power up pre-commit [\#237](https://github.com/python-adaptive/adaptive/pull/237) ([basnijholt](https://github.com/basnijholt))
- add a section of "How to cite" Adaptive [\#235](https://github.com/python-adaptive/adaptive/pull/235) ([basnijholt](https://github.com/basnijholt))
- Fix SKOptLearner for multi variate domain \(issue \#233\) [\#234](https://github.com/python-adaptive/adaptive/pull/234) ([caenrigen](https://github.com/caenrigen))
- add a time-base stopping criterion for runners [\#229](https://github.com/python-adaptive/adaptive/pull/229) ([jbweston](https://github.com/jbweston))
- update packages in tutorial's landing page [\#224](https://github.com/python-adaptive/adaptive/pull/224) ([basnijholt](https://github.com/basnijholt))
- add \_RequireAttrsABCMeta and make the BaseLearner use it [\#222](https://github.com/python-adaptive/adaptive/pull/222) ([basnijholt](https://github.com/basnijholt))
- 2D: add triangle\_loss [\#221](https://github.com/python-adaptive/adaptive/pull/221) ([basnijholt](https://github.com/basnijholt))
- 2D: add interpolated\_on\_grid method [\#216](https://github.com/python-adaptive/adaptive/pull/216) ([basnijholt](https://github.com/basnijholt))
- WIP: raise an error when using a lambda and default executor [\#210](https://github.com/python-adaptive/adaptive/pull/210) ([basnijholt](https://github.com/basnijholt))

## [v0.10.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.10.0-dev) (2019-10-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.9.0...v0.10.0-dev)

**Closed issues:**

- Command-line tool [\#217](https://github.com/python-adaptive/adaptive/issues/217)
- release v0.9.0 [\#212](https://github.com/python-adaptive/adaptive/issues/212)

## [v0.9.0](https://github.com/python-adaptive/adaptive/tree/v0.9.0) (2019-09-23)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.8.1...v0.9.0)

**Implemented enhancements:**

- jupyter-sphinx update [\#203](https://github.com/python-adaptive/adaptive/issues/203)

**Closed issues:**

- jupyter-sphinx is pinned to non-existing branch [\#199](https://github.com/python-adaptive/adaptive/issues/199)

**Merged pull requests:**

- pass value\_scale to the LearnerND's loss\_per\_simplex function [\#219](https://github.com/python-adaptive/adaptive/pull/219) ([basnijholt](https://github.com/basnijholt))
- add scatter\_or\_line argument to Learner1D.plot [\#215](https://github.com/python-adaptive/adaptive/pull/215) ([basnijholt](https://github.com/basnijholt))
- remove MPI4PY\_MAX\_WORKERS where it's not used [\#209](https://github.com/python-adaptive/adaptive/pull/209) ([basnijholt](https://github.com/basnijholt))
- use jupyter\_sphinx v0.2.0 from conda instead of my branch [\#204](https://github.com/python-adaptive/adaptive/pull/204) ([basnijholt](https://github.com/basnijholt))
- Authors [\#202](https://github.com/python-adaptive/adaptive/pull/202) ([basnijholt](https://github.com/basnijholt))
- Update tutorial.parallelism.rst [\#201](https://github.com/python-adaptive/adaptive/pull/201) ([aeantipov](https://github.com/aeantipov))
- ensure atomic writes when saving a file [\#200](https://github.com/python-adaptive/adaptive/pull/200) ([basnijholt](https://github.com/basnijholt))
- don't treat the no data case differently in the Learner1D [\#195](https://github.com/python-adaptive/adaptive/pull/195) ([basnijholt](https://github.com/basnijholt))
- pin everything in the docs/environment.yml file [\#194](https://github.com/python-adaptive/adaptive/pull/194) ([basnijholt](https://github.com/basnijholt))
- Add a SequenceLearner [\#193](https://github.com/python-adaptive/adaptive/pull/193) ([basnijholt](https://github.com/basnijholt))
- Use black for code formatting [\#191](https://github.com/python-adaptive/adaptive/pull/191) ([basnijholt](https://github.com/basnijholt))
- BalancingLearner: add a "cycle" strategy, sampling the learners one by one [\#188](https://github.com/python-adaptive/adaptive/pull/188) ([basnijholt](https://github.com/basnijholt))

## [v0.8.1](https://github.com/python-adaptive/adaptive/tree/v0.8.1) (2019-05-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.9.0-dev...v0.8.1)

**Closed issues:**

- release v0.8.0 [\#165](https://github.com/python-adaptive/adaptive/issues/165)

## [v0.9.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.9.0-dev) (2019-05-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.8.0...v0.9.0-dev)

## [v0.8.0](https://github.com/python-adaptive/adaptive/tree/v0.8.0) (2019-05-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.6...v0.8.0)

**Implemented enhancements:**

- set python\_requires in setup.py [\#179](https://github.com/python-adaptive/adaptive/issues/179)

**Fixed bugs:**

- Learner2D.plot\(\) returns NaN [\#181](https://github.com/python-adaptive/adaptive/issues/181)
- Runner errors for Python 3.7 when done [\#156](https://github.com/python-adaptive/adaptive/issues/156)
- 2D: do not return points outside the bounds, closes \#181 [\#182](https://github.com/python-adaptive/adaptive/pull/182) ([basnijholt](https://github.com/basnijholt))

**Closed issues:**

- default loss of LearnerND changed? [\#171](https://github.com/python-adaptive/adaptive/issues/171)
- Add a page to the documentation of papers where adaptive is used [\#163](https://github.com/python-adaptive/adaptive/issues/163)

**Merged pull requests:**

- add support for mpi4py [\#190](https://github.com/python-adaptive/adaptive/pull/190) ([basnijholt](https://github.com/basnijholt))
- use pytest rather than py.test [\#189](https://github.com/python-adaptive/adaptive/pull/189) ([basnijholt](https://github.com/basnijholt))
- renormalize the plots value axis on every update [\#186](https://github.com/python-adaptive/adaptive/pull/186) ([basnijholt](https://github.com/basnijholt))
- use setup\(..., python\_requires='\>=3.6'\), closes \#179 [\#180](https://github.com/python-adaptive/adaptive/pull/180) ([basnijholt](https://github.com/basnijholt))
- use the repo code in docs execute [\#177](https://github.com/python-adaptive/adaptive/pull/177) ([basnijholt](https://github.com/basnijholt))
- do not inline the HoloViews JS [\#176](https://github.com/python-adaptive/adaptive/pull/176) ([basnijholt](https://github.com/basnijholt))
- add a gallery page of Adaptive uses in scientific works [\#174](https://github.com/python-adaptive/adaptive/pull/174) ([basnijholt](https://github.com/basnijholt))
- LearnerND: change the required loss to 1e-3 because the loss definition changed [\#172](https://github.com/python-adaptive/adaptive/pull/172) ([basnijholt](https://github.com/basnijholt))
- Add logo to the documentation [\#170](https://github.com/python-adaptive/adaptive/pull/170) ([basnijholt](https://github.com/basnijholt))
- import Iterable and Sized from collections.abc [\#161](https://github.com/python-adaptive/adaptive/pull/161) ([basnijholt](https://github.com/basnijholt))
- invoke conda directly in CI [\#153](https://github.com/python-adaptive/adaptive/pull/153) ([basnijholt](https://github.com/basnijholt))
- change urls from GitLab to GitHub [\#146](https://github.com/python-adaptive/adaptive/pull/146) ([basnijholt](https://github.com/basnijholt))

## [v0.7.6](https://github.com/python-adaptive/adaptive/tree/v0.7.6) (2019-03-21)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.5...v0.7.6)

**Fixed bugs:**

- live\_plot broken with latest holoviews and bokeh [\#166](https://github.com/python-adaptive/adaptive/issues/166)

**Merged pull requests:**

- do not force shutdown the executor in the cleanup [\#164](https://github.com/python-adaptive/adaptive/pull/164) ([basnijholt](https://github.com/basnijholt))

## [v0.7.5](https://github.com/python-adaptive/adaptive/tree/v0.7.5) (2019-03-19)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.4...v0.7.5)

**Fixed bugs:**

- BalancingLearner puts all points in the first child-learner when asking for points with no data present [\#159](https://github.com/python-adaptive/adaptive/issues/159)

**Merged pull requests:**

- fix live\_plot [\#167](https://github.com/python-adaptive/adaptive/pull/167) ([basnijholt](https://github.com/basnijholt))

## [v0.7.4](https://github.com/python-adaptive/adaptive/tree/v0.7.4) (2019-02-15)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.3...v0.7.4)

**Fixed bugs:**

- Loading data file with no data results in an error for the BalancingLearner  [\#148](https://github.com/python-adaptive/adaptive/issues/148)
- Returning np.nan breaks the 1D learner [\#145](https://github.com/python-adaptive/adaptive/issues/145)
- fix bug in curvature\_loss [\#152](https://github.com/python-adaptive/adaptive/pull/152) ([basnijholt](https://github.com/basnijholt))

**Merged pull requests:**

- handle NaN losses and add a test, closes \#145 [\#151](https://github.com/python-adaptive/adaptive/pull/151) ([basnijholt](https://github.com/basnijholt))
- handle empty data files when loading, closes \#148 [\#149](https://github.com/python-adaptive/adaptive/pull/149) ([basnijholt](https://github.com/basnijholt))

## [v0.7.3](https://github.com/python-adaptive/adaptive/tree/v0.7.3) (2019-01-29)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.2...v0.7.3)

**Implemented enhancements:**

- Add a sequential executor [\#138](https://github.com/python-adaptive/adaptive/issues/138)
- Add tests for 1D interpolator learner [\#136](https://github.com/python-adaptive/adaptive/issues/136)
- Add integration learner [\#135](https://github.com/python-adaptive/adaptive/issues/135)
- Make the runner work with `asyncio` and inside Jupyter notebooks [\#133](https://github.com/python-adaptive/adaptive/issues/133)
- Add module for notebook integration and shortcuts for common executors [\#132](https://github.com/python-adaptive/adaptive/issues/132)
- Add homogeneous sampling learner [\#131](https://github.com/python-adaptive/adaptive/issues/131)
- Add a "balancing" learner [\#130](https://github.com/python-adaptive/adaptive/issues/130)
- Implement 2D and 3D learners [\#129](https://github.com/python-adaptive/adaptive/issues/129)
- Add a 0D averaging learner [\#128](https://github.com/python-adaptive/adaptive/issues/128)
- Write `interpolate` for the 1D learner such that it is more efficient [\#126](https://github.com/python-adaptive/adaptive/issues/126)
- Gracefully handle exceptions when evaluating the function to be learned [\#125](https://github.com/python-adaptive/adaptive/issues/125)
- Allow BalancingLearner to return arbitrary number of points from 'choose\_points' [\#124](https://github.com/python-adaptive/adaptive/issues/124)
- Increase the default refresh rate for 'live\_plot' [\#120](https://github.com/python-adaptive/adaptive/issues/120)
- remove default number of points to choose in `choose\_points` [\#118](https://github.com/python-adaptive/adaptive/issues/118)
- Consider using Gaussian process optimization as a learner [\#115](https://github.com/python-adaptive/adaptive/issues/115)
- Make `distributed.Client` work with automatic scaling of the cluster [\#104](https://github.com/python-adaptive/adaptive/issues/104)
- Improve plotting for learners [\#83](https://github.com/python-adaptive/adaptive/issues/83)
- \(refactor\) learner.tell\(x, None\) might be renamed to learner.tell\_pending\(x\) [\#73](https://github.com/python-adaptive/adaptive/issues/73)
- \(feature\) make interactive plots for learnerND plot\_slice method [\#64](https://github.com/python-adaptive/adaptive/issues/64)
- \(LearnerND\) make default loss function better [\#63](https://github.com/python-adaptive/adaptive/issues/63)
- allow for N-d output [\#60](https://github.com/python-adaptive/adaptive/issues/60)
- add cross-section plot [\#58](https://github.com/python-adaptive/adaptive/issues/58)
- \(BalancingLearner\) make new balancinglearner that looks at the total loss rather than loss improvement [\#36](https://github.com/python-adaptive/adaptive/issues/36)
- \(LearnerND\) allow any convex hull as domain [\#25](https://github.com/python-adaptive/adaptive/issues/25)
- \(Learner1D\) add possibility to use the direct neighbors in the loss [\#20](https://github.com/python-adaptive/adaptive/issues/20)

**Fixed bugs:**

- Distinguish actual loss and estimated loss [\#139](https://github.com/python-adaptive/adaptive/issues/139)
- Set the bounds in a smarter way [\#127](https://github.com/python-adaptive/adaptive/issues/127)
- some points get cluttered [\#86](https://github.com/python-adaptive/adaptive/issues/86)
- 2D learner specifies a 1D point causing 2D learner to fail [\#81](https://github.com/python-adaptive/adaptive/issues/81)
- Method 'Learner.tell' is ambiguous [\#80](https://github.com/python-adaptive/adaptive/issues/80)
- Learner1D fails with extremely narrow features [\#78](https://github.com/python-adaptive/adaptive/issues/78)
- AverageLearner math domain error [\#77](https://github.com/python-adaptive/adaptive/issues/77)
- \(LearnerND\) scale y-values [\#61](https://github.com/python-adaptive/adaptive/issues/61)
- Learner1D breaks if right bound is added before the left bound [\#45](https://github.com/python-adaptive/adaptive/issues/45)
- Learner1D's bound check algo in self.ask doesn't take self.data or self.pending\_points [\#44](https://github.com/python-adaptive/adaptive/issues/44)
- Learner1D fails when function returns a list instead of a numpy.array [\#43](https://github.com/python-adaptive/adaptive/issues/43)
- Learner1D fails when a point \(x, None\) is added when x already exists [\#42](https://github.com/python-adaptive/adaptive/issues/42)
- Learner1D.ask breaks when adding points in some order [\#41](https://github.com/python-adaptive/adaptive/issues/41)
- Learner1D doesn't correctly set the interpolated loss when a point is added [\#40](https://github.com/python-adaptive/adaptive/issues/40)
- Learner1D could in some situations return -inf as loss improvement, which would make balancinglearner never choose to improve [\#35](https://github.com/python-adaptive/adaptive/issues/35)
- LearnerND fails for BalancingLearner test [\#34](https://github.com/python-adaptive/adaptive/issues/34)
- Learner2D suggests same point twice [\#30](https://github.com/python-adaptive/adaptive/issues/30)
- \(LearnerND\) if you stop the runner, and then try to continue, it fails. [\#23](https://github.com/python-adaptive/adaptive/issues/23)

**Closed issues:**

- Add Authors file and review license [\#137](https://github.com/python-adaptive/adaptive/issues/137)
- make the runner request points until it's using all cores [\#123](https://github.com/python-adaptive/adaptive/issues/123)
- Remove \_choose\_points [\#121](https://github.com/python-adaptive/adaptive/issues/121)
- Fix extremely long kernel restart times [\#119](https://github.com/python-adaptive/adaptive/issues/119)
- live plotting: add a universal visual cue that the goal is achieved. [\#117](https://github.com/python-adaptive/adaptive/issues/117)
- ipyparallel shouldn't be a dependency [\#114](https://github.com/python-adaptive/adaptive/issues/114)
- adaptive fails to discover features [\#113](https://github.com/python-adaptive/adaptive/issues/113)
- add tests for 2D learner [\#111](https://github.com/python-adaptive/adaptive/issues/111)
- DataSaver doesn't work with the BalancingLearner [\#110](https://github.com/python-adaptive/adaptive/issues/110)
- deleted issue [\#108](https://github.com/python-adaptive/adaptive/issues/108)
- removing optional dependencies [\#106](https://github.com/python-adaptive/adaptive/issues/106)
- Improve ipython event loop integration [\#105](https://github.com/python-adaptive/adaptive/issues/105)
- Use holoviews.TriMesh when it makes it to a release [\#103](https://github.com/python-adaptive/adaptive/issues/103)
- save live plots into internal datastructure [\#101](https://github.com/python-adaptive/adaptive/issues/101)
- To-dos before making the repo public [\#100](https://github.com/python-adaptive/adaptive/issues/100)
- set the correct loss\_improvement for the AverageLearner [\#95](https://github.com/python-adaptive/adaptive/issues/95)
- Ensure a minimum resolution [\#92](https://github.com/python-adaptive/adaptive/issues/92)
- change the error message in runner [\#91](https://github.com/python-adaptive/adaptive/issues/91)
- The ProcessPoolExecutor doesn't work on Windows [\#90](https://github.com/python-adaptive/adaptive/issues/90)
- 1D and 2D learner: stop interpolating function instead of the loss [\#87](https://github.com/python-adaptive/adaptive/issues/87)
- Discontinuities in zero should be detected and be approximated with some margin [\#85](https://github.com/python-adaptive/adaptive/issues/85)
- \(minor bug\) learner.choose\_points gives wrong number of points in one very particular case [\#84](https://github.com/python-adaptive/adaptive/issues/84)
- 2D: if boundary point fails it will never be re-evaluated ... [\#82](https://github.com/python-adaptive/adaptive/issues/82)
- Learner2D + BalancingLearner too slow to use on many cores [\#79](https://github.com/python-adaptive/adaptive/issues/79)
- BalancingLearner.from\_product doesn't work with the DataSaver [\#74](https://github.com/python-adaptive/adaptive/issues/74)
- Follow-up from "WIP: Add LearnerND that does not interpolate the values of pending points" [\#70](https://github.com/python-adaptive/adaptive/issues/70)
- \(triangulation\) make method for finding initial simplex part of the triangulation class [\#68](https://github.com/python-adaptive/adaptive/issues/68)
- \(refactor\) LearnerND.\_ask can be refactored to be so much more readable [\#67](https://github.com/python-adaptive/adaptive/issues/67)
- \(LearnerND\) make choose point in simplex better [\#62](https://github.com/python-adaptive/adaptive/issues/62)
- Make learnerND datastructures immutable where possible [\#54](https://github.com/python-adaptive/adaptive/issues/54)
- Rename LearnerND to TriangulatingLearner [\#51](https://github.com/python-adaptive/adaptive/issues/51)
- tell\_many method [\#49](https://github.com/python-adaptive/adaptive/issues/49)
- Set up documentation [\#48](https://github.com/python-adaptive/adaptive/issues/48)
- DeprecationWarning: sorted\_dict.iloc is deprecated. Use SortedDict.keys\(\) instead. [\#47](https://github.com/python-adaptive/adaptive/issues/47)
- The example given in data\_saver.py doesn't compile. [\#46](https://github.com/python-adaptive/adaptive/issues/46)
- What should learners do when fed the same point twice [\#39](https://github.com/python-adaptive/adaptive/issues/39)
- How should learners handle data that is outside of the domain [\#38](https://github.com/python-adaptive/adaptive/issues/38)
- No tests for the 'BalancingLearner' [\#37](https://github.com/python-adaptive/adaptive/issues/37)
- release 0.6.0 [\#33](https://github.com/python-adaptive/adaptive/issues/33)
- Make BaseRunner an abstract base class [\#32](https://github.com/python-adaptive/adaptive/issues/32)
- \(BalancingLearner\) loss is cached incorrectly [\#31](https://github.com/python-adaptive/adaptive/issues/31)
- LearnerND triangulation incomplete [\#29](https://github.com/python-adaptive/adaptive/issues/29)
- \(LearnerND\) flat simplices are sometimes added on the surface of the triangulation [\#28](https://github.com/python-adaptive/adaptive/issues/28)
- \(LearnerND\) add iso-surface plot feature [\#27](https://github.com/python-adaptive/adaptive/issues/27)
- make BalancingLearner work with the live\_plot [\#26](https://github.com/python-adaptive/adaptive/issues/26)
- test\_balancing\_learner\[Learner2D-ring\_of\_fire-learner\_kwargs2\] fails sometimes [\#24](https://github.com/python-adaptive/adaptive/issues/24)
- widgets don't show up on adaptive.readthedocs.io [\#21](https://github.com/python-adaptive/adaptive/issues/21)
- How to handle NaN? [\#18](https://github.com/python-adaptive/adaptive/issues/18)
- Remove public 'fname' learner attribute [\#17](https://github.com/python-adaptive/adaptive/issues/17)
- Release v0.7.0 [\#14](https://github.com/python-adaptive/adaptive/issues/14)
- \(Learner1D\) improve time complexity [\#13](https://github.com/python-adaptive/adaptive/issues/13)
- Typo in documentation for` adaptive.learner.learner2D.uniform\_loss\(ip\)` [\#12](https://github.com/python-adaptive/adaptive/issues/12)
- \(LearnerND\) fix plotting of scaled domains [\#11](https://github.com/python-adaptive/adaptive/issues/11)
- suggested points lie outside of domain [\#7](https://github.com/python-adaptive/adaptive/issues/7)
- DEVELOPMENT IS ON GITLAB: https://gitlab.kwant-project.org/qt/adaptive [\#5](https://github.com/python-adaptive/adaptive/issues/5)

**Merged pull requests:**

- fix \_get\_data for the BalancingLearner [\#150](https://github.com/python-adaptive/adaptive/pull/150) ([basnijholt](https://github.com/basnijholt))

## [v0.7.2](https://github.com/python-adaptive/adaptive/tree/v0.7.2) (2018-12-17)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.1...v0.7.2)

## [v0.7.1](https://github.com/python-adaptive/adaptive/tree/v0.7.1) (2018-12-17)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.8.0-dev...v0.7.1)

## [v0.8.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.8.0-dev) (2018-12-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.0...v0.8.0-dev)

## [v0.7.0](https://github.com/python-adaptive/adaptive/tree/v0.7.0) (2018-12-07)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.7.0-dev...v0.7.0)

**Closed issues:**

- gif in the README [\#1](https://github.com/python-adaptive/adaptive/issues/1)

## [v0.7.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.7.0-dev) (2018-10-09)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.6.0...v0.7.0-dev)

## [v0.6.0](https://github.com/python-adaptive/adaptive/tree/v0.6.0) (2018-10-01)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.5.0...v0.6.0)

## [v0.5.0](https://github.com/python-adaptive/adaptive/tree/v0.5.0) (2018-08-20)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.4.1...v0.5.0)

**Closed issues:**

- Issue using distributed [\#3](https://github.com/python-adaptive/adaptive/issues/3)

## [v0.4.1](https://github.com/python-adaptive/adaptive/tree/v0.4.1) (2018-05-28)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.5.0-dev...v0.4.1)

## [v0.5.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.5.0-dev) (2018-05-24)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.4.0...v0.5.0-dev)

## [v0.4.0](https://github.com/python-adaptive/adaptive/tree/v0.4.0) (2018-05-24)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.4.0-dev...v0.4.0)

## [v0.4.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.4.0-dev) (2018-04-12)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.3.0...v0.4.0-dev)

## [v0.3.0](https://github.com/python-adaptive/adaptive/tree/v0.3.0) (2018-03-28)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.2.1...v0.3.0)

## [v0.2.1](https://github.com/python-adaptive/adaptive/tree/v0.2.1) (2018-03-03)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.3.0-dev...v0.2.1)

## [v0.3.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.3.0-dev) (2018-02-23)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.2.0...v0.3.0-dev)

## [v0.2.0](https://github.com/python-adaptive/adaptive/tree/v0.2.0) (2018-02-23)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.2.0-dev...v0.2.0)

## [v0.2.0-dev](https://github.com/python-adaptive/adaptive/tree/v0.2.0-dev) (2018-02-21)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/v0.1.0...v0.2.0-dev)

## [v0.1.0](https://github.com/python-adaptive/adaptive/tree/v0.1.0) (2018-02-21)

[Full Changelog](https://github.com/python-adaptive/adaptive/compare/03236d4aa3919dbc469f26d4925ed5097b1e4a04...v0.1.0)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
