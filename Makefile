quality_cheks:
	isort ./services/training_service
	isort ./services/prediction_service
	isort ./services/evidently_service

	black ./services/training_service
	black ./services/prediction_service
	black ./services/evidently_service

	pylint --recursive=y ./services/training_service
	pylint --recursive=y ./services/prediction_service
	pylint --recursive=y ./services/evidently_service

prepare_db:
	rm ./services/training_service/models/mlflow.db
	echo databased removed
	touch ./services/training_service/models/mlflow.db
	echo database is cleaned




build-docker-containers:
	 docker-compose up --force-recreate --build


build : quality_cheks prepare_db build-docker-containers
