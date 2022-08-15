prepare_env:

quality_cheks:
	pipenv shell
	isort ./services/training_service
	isort ./services/prediction_service
	isort ./services/evidently_service

	black ./services/training_service
	black ./services/prediction_service
	black ./services/evidently_service

	pylint --recursive=y ./services/training_service
	pylint --recursive=y ./services/prediction_service
	pylint --recursive=y ./services/prediction_service

prepare_db:
	rm ./services/training_service/models/mlflow.db
	echo databased removed
	touch ./services/training_service/models/mlflow.db
	echo database is cleaned




build-docker-containers:
	 docker-compose up


build : prepare_env quality_cheks prepare_db build-docker-containers
