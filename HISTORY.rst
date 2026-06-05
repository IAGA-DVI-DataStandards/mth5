Changelog
=========

All notable changes to this project will be documented in this file.
Changelog
=========

All notable changes to this project will be documented in this file.
Changelog
=========

All notable changes to this project will be documented in this file.
Changelog
=========

All notable changes to this project will be documented in this file.
History
=========

0.1.0 (2021-06-30)
------------------

* First release on PyPI.

0.2.0 (2021-10-31)
-------------------

* Updated the structure of MTH5 to have ``Experiment`` as the top level
* Updated tests
* Backwards compatibility works
* Updated Docs

0.2.5 (2022-04-07)
----------------------

* fixed bugs
* Added TransferFunctions and TransferFunction groups at the station level that can now hold transfer functions
* Added channel_summary and tf_summary tables that are updated upon close if the file is in 'w' mode
* Updated tests
* Updated documentation
* Note: tests for make_mth5 from IRIS are currently not working as there has been some reorganization with data at the DMC

0.2.6 (2022-07-01)
-----------------------

* Added calibration functions
* minor bug fixes
* updated tests
* updated documentation

0.2.7 (2022-09-14)
------------------------

* Rebased IO module to make a module for each data logger type
* Updated tests
* Updated documentation
* Factored `make_mth5`

0.3.0 (2022-09-25)
------------------------

* change default initialize_mth5 to use append mode, issue #92 by @kkappler in #94
* Fix issue 105 by @kkappler in PR #106
* adding in parallel mth5 tutorial by @nre900 in #110
* adding in new tutorial and modifications to mth5_in_parallel.ipynb by @nre900 in issue #112
* Add phoenix reader by @kujaku11 in PR #103
* Remove response by @kujaku11 in PR #100 

0.3.1 (2023-01-18)
------------------------

* Speed up station and survey validataion by 
* Tutorial updates by @nre900 
* remove kwarg specifying default value 
* update initialize_mth5 
* Have a single metadata object for ChannelTS and RunTS 
* Use h5 Paths to get groups and datasets
* Bump wheel from 0.33.6 to 0.38.1

0.4.0 (2023-09-29)
------------------------

* Adding __add__ and merge to ChannelTS in https://github.com/kujaku11/mth5/pull/136
* Added Clients in https://github.com/kujaku11/mth5/pull/82 incuding USGS Geomagnetic client
* Add Scipy filters to xarray in https://github.com/kujaku11/mth5/pull/139
* Fix FDSN download with no channel filters in https://github.com/kujaku11/mth5/pull/137
* Align Channels in RunTS in https://github.com/kujaku11/mth5/pull/141
* Fix FDSN download without all runs in https://github.com/kujaku11/mth5/pull/154
* Add ability to store Fouried Coefficients estimated by processing software
* Updated Phoenix reader
* Using Loguru now instead of built-in logging module
* Fixed slicing of channel dataset

0.4.1 (2023-11-08)
-------------------------

* minor bug fixes
* removed restriction on pandas < 2

0.4.3
-------------------------

* Fix issue 171 by @kkappler in https://github.com/kujaku11/mth5/pull/172
* Add try except for runs with no channels by @kkappler in https://github.com/kujaku11/mth5/pull/176
* remove references to filter correction operation by @kkappler in https://github.com/kujaku11/mth5/pull/177
* Fix mt metadata issue 173 by @kkappler in https://github.com/kujaku11/mth5/pull/174
* Notebook updates by @kujaku11 in https://github.com/kujaku11/mth5/pull/178
* add (failing) test that fc_metadata updates as expected by @kkappler in https://github.com/kujaku11/mth5/pull/173
* Add return self to open_mth5() by @kkappler in https://github.com/kujaku11/mth5/pull/183
* Patches by @kkappler in https://github.com/kujaku11/mth5/pull/181
* Patches by @kkappler in https://github.com/kujaku11/mth5/pull/184
* Update how MTH5Tables handles dtype by @kujaku11 in https://github.com/kujaku11/mth5/pull/192
* Patches by @kkappler in https://github.com/kujaku11/mth5/pull/190
* add synthetic time series by @kkappler in https://github.com/kujaku11/mth5/pull/194
* Patches by @kujaku11 in https://github.com/kujaku11/mth5/pull/195
* Update tests.yml by @kujaku11 in https://github.com/kujaku11/mth5/pull/196
* Run ipynb on tests by @kujaku11 in https://github.com/kujaku11/mth5/pull/200

0.4.5
------------------------

* Wtf by @kujaku11 in https://github.com/kujaku11/mth5/pull/205
* Fix issue 191 by @kkappler in https://github.com/kujaku11/mth5/pull/208
* Fix issue 191 by @kkappler in https://github.com/kujaku11/mth5/pull/211
* Fix issue 209 by @kkappler in https://github.com/kujaku11/mth5/pull/210
* Update station.py by @kujaku11 in https://github.com/kujaku11/mth5/pull/215
* Patches by @kkappler in https://github.com/kujaku11/mth5/pull/206
* Fix issue 219 by @kkappler in https://github.com/kujaku11/mth5/pull/222
* Fix issue 217 by @kkappler in https://github.com/kujaku11/mth5/pull/218
* try fix #223 by @kkappler in https://github.com/kujaku11/mth5/pull/224
* Add some more multivariate functionality by @kkappler in https://github.com/kujaku11/mth5/pull/228
* Fix issue 209a by @kkappler in https://github.com/kujaku11/mth5/pull/231
* Minor multivariate updates by @kkappler in https://github.com/kujaku11/mth5/pull/232
* Updating bugs and Adding functionality by @kujaku11 in https://github.com/kujaku11/mth5/pull/226
* bump v0.4.3 --> v0.4.4 by @kujaku11 in https://github.com/kujaku11/mth5/pull/216

0.4.6 (2024-08-16)
----------------------------

* add aurora tools to mth5 by @kujaku11 in https://github.com/kujaku11/mth5/pull/229
* Patches by @kkappler in https://github.com/kujaku11/mth5/pull/234
* Fix issue 233 by @kkappler in https://github.com/kujaku11/mth5/pull/235
* hotfix synthetic electric field polarity (#236) by @kkappler in https://github.com/kujaku11/mth5/pull/237

0.4.7 (2024-09-30)
---------------------------

* Fix issue 191 by @kkappler in https://github.com/kujaku11/mth5/pull/239
* Update synthetic data by @kkappler in https://github.com/kujaku11/mth5/pull/243
* Optimize adding TF by @kujaku11 in https://github.com/kujaku11/mth5/pull/242
* Patches by @kujaku11 in https://github.com/kujaku11/mth5/pull/244
* bump v0.4.6 --> v0.4.7 by @kujaku11 in https://github.com/kujaku11/mth5/pull/245

0.4.8 (2024-10-14)
-------------------------

* Fix Issue 241 by @dequiroga in https://github.com/kujaku11/mth5/pull/246
* Updating clients by @kujaku11 in https://github.com/kujaku11/mth5/pull/249
* Update MakeMTH5 and clients by @kujaku11 in https://github.com/kujaku11/mth5/pull/189
* Updating Clients by @kujaku11 in https://github.com/kujaku11/mth5/pull/250
* @dequiroga made their first contribution in https://github.com/kujaku11/mth5/pull/246

**Full Changelog**: https://github.com/kujaku11/mth5/compare/v0.4.7...v0.4.8

0.4.9
-----------------------------

* Update documentation by @kujaku11 in https://github.com/kujaku11/mth5/pull/251
* Fix issue 252 by @kkappler in https://github.com/kujaku11/mth5/pull/253

**Full Changelog**: https://github.com/kujaku11/mth5/compare/v0.4.8...v0.4.9

0.4.10
------------------------------

* HOT FIX: importing scipy, to use scipy.stats by @dequiroga in https://github.com/kujaku11/mth5/pull/258
* FC dataset end time consistency by @dequiroga in https://github.com/kujaku11/mth5/pull/264
* Add Metronix reader by @kujaku11 in https://github.com/kujaku11/mth5/pull/261
* Resolve SyntaxWarnings in Python 3.12 and above for invalid escape sequences by @jameswilburlewis in https://github.com/kujaku11/mth5/pull/256
* Patches by @kujaku11 in https://github.com/kujaku11/mth5/pull/265
* Update README.md by @kujaku11 in https://github.com/kujaku11/mth5/pull/266

0.5.0 (2024-10-25)
------------------------------

0.6.0 (2026-01-20)
------------------------------

* Pydantic optimize tests by @kujaku11 in https://github.com/kujaku11/mth5/pull/305
* updates to run_ts by @kkappler in https://github.com/kujaku11/mth5/pull/307
* Adding legacy phoenix MTU reader by @kujaku11 in https://github.com/kujaku11/mth5/pull/308
* Lemi calibrations by @kujaku11 in https://github.com/kujaku11/mth5/pull/309
* NIMS Holder  by @jpopelar in https://github.com/kujaku11/mth5/pull/310
* Pydantic updates and transition to pytest by @kujaku11 in https://github.com/kujaku11/mth5/pull/306
* Update code base to run with mt-metadata v1.0 and translate to pytest. by @kujaku11 in https://github.com/kujaku11/mth5/pull/311

0.6.1 (2026-01-26)
------------------------------

* 2026-01-20 — PR #311 — Improve PhoenixClient calibration handling and docs
* 2026-01-20 — PR #306 — Pydantic updates and transition to pytest


**Full Changelog**: https://github.com/kujaku11/mth5/compare/v0.6.0...v0.6.1Changelog All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.All notable changes to this project will be documented in this
file.
