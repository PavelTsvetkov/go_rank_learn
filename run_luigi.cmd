rem activate kgs_learn
set PYTHONPATH=%cd%
luigi --module luigi_tasks GenerateRawTrainingData --local-scheduler