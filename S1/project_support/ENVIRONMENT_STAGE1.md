# Stage 1 computational environment

This is a historical environment record. Use the package-level dependency lock and README when setting up the shared replication package; original workspace paths below describe the earlier setup.

A project-local `.venv` was created with the existing Anaconda Python 3.13.5, leaving the user-specified `environment.yml` unchanged. Installed versions are recorded in `requirements-stage1.lock.txt`; major packages include econml 0.17.0, scikit-learn 1.9.0, pandas 3.0.5, and numpy 2.5.3. These versions describe the replication environment.

From the project root, run Python scripts with `.venv/bin/python`. In the IDE, select this interpreter for new estimation notebooks. Existing data/audit notebooks were verified using the already available notebook kernel; each notebook's execution scope is documented.

Recreate the local environment with Python 3.13.5 and `python -m pip install -r requirements-stage1.lock.txt`. The freeze records the resolved dependencies on this macOS machine; it does not promise portability to every platform.

A synthetic CausalForestDML environment check passed: known simulated effect 0.5, estimated 0.484, interval [0.363, 0.605]. This verifies basic execution, not scientific validity or TEDS results. Reproduce it with `notebooks/S1/helpers/verify_estimation_environment.py`.

Categorical imputation continues to use R 4.6.0 and project-local mice 3.19.0 under `.r-library`. The corrected augmentation helper is guarded to this mice version because it adapts an internal implementation. Per-fit convergence checks and the information-boundary tests remain necessary.
