# Environment
KERNEL_NAME="JAMIE"

# Results
# jupyter nbconvert --execute --to notebook --inplace sample.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
# jupyter nbconvert --execute --to notebook --inplace scGEM.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
# jupyter nbconvert --execute --to notebook --inplace MMD-MA.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
# jupyter nbconvert --execute --to notebook --inplace scMNC-Motor.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
# jupyter nbconvert --execute --to notebook --inplace scMNC-Visual.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
jupyter nbconvert --execute --to notebook --inplace brainchromatin.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
jupyter nbconvert --execute --to notebook --inplace DM_rep4-Imp.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\

# Revisions
jupyter nbconvert --execute --to notebook --inplace scMNC-Visual-Tuning.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
jupyter nbconvert --execute --to notebook --inplace scMNC-Visual-Cortical.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
jupyter nbconvert --execute --to notebook --inplace Simulation-1250.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
jupyter nbconvert --execute --to notebook --inplace scGLUE.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME
# jupyter nbconvert --execute --to notebook --inplace time-and-memory.ipynb --ExecutePreprocessor.kernel_name=$KERNEL_NAME ;\
