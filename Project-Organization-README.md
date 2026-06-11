BookHead-AI-App
==============================

AI based web application for book readers and librarians. 
Helps in deciding next read based on reading preference and available books in the shelf.
Helps librarian to create a comprehensive book catalog in seconds.

Project Organization
------------

    ├── LICENSE
    ├── README.md                       <- The top-level README for developers using this project.
    │
    ├── main.py                         <- Script for starting FastAPI service and app startup
    │
    ├── models                          <- Serialized models used across the application
    │
    ├── app                             <- application source code
    │   ├── api                         <- Scripts for FastAPI configuration and endpoints
    │   |   └── routes                  <- Code for routes configuration and endpoints
    │   |          ├── reader.py        <- Code for all reader routes
    │   |          └── librarian.py     <- Code for all librarian routes
    |   |
    │   ├── core                        <- Scripts for app core configuration and settings
    │   |   ├── celery_app.py           <- Code for configuring celery workers
    │   |   ├── config.py               <- Code for settings variables used across the application
    │   |   ├── exceptions.py           <- Code for custom exceptions
    │   |   ├── logging.py              <- Code for configuring app logging
    │   |   └── worker_services.py      <- Code for wiring up services, utilised by celery workers
    |   |
    │   ├── models                      <- Scripts for Pydantic models used across the application
    │   |   ├── book.py                 <- Code for book models
    │   |   ├── request.py              <- Code for API request models
    │   |   └── response.py             <- Code for API response models
    |   |
    │   ├── pipelines                   <- Scripts for inference pipelines used after user input
    │   |   ├── reader_pipeline.py      <- Script for reader pipeline workflow
    │   |   └── librarian_pipeline.py   <- Script for librarian pipeline workflow
    |   |
    │   ├── services                    <- Scripts for services used across the app
    │   |   ├── base.py                 <- Code for Abstract Base Class for every service
    │   |   ├── catalog.py              <- Code for librarian catalog generation
    │   |   ├── confidence_router.py    <- Code for routing OCR results based on confidence score
    │   |   ├── crop_store.py           <- Code for storing cropped book spine images in disk
    │   |   ├── detection.py            <- Code for detecting, storing, running OCR on book spines and 
    |   |   |                              validating OCR results and extracting meta-data from Google
    |   |   |                              book API
    │   |   ├── embeddings.py           <- Code for creating vector embeddings using embedding model
    │   |   ├── image_saver.py          <- Code for saving orginal user input in disk
    │   |   ├── image_utils.py          <- Code for utilities used for image processing
    │   |   ├── job_store.py            <- Code for redis interactions - complete CRUD operations
    │   |   ├── llm_client.py           <- Code for configuring LLM clients (both API and in-system)
    │   |   ├── llm_parser.py           <- Code for parsing LLM response
    │   |   ├── model_loader.py         <- Code for loading Vision model for book spine detection
    │   |   ├── ocr_preprocessing.py    <- Code for preprocessing images before OCR
    │   |   ├── ocr.py                  <- Code for running OCR pipeline - Text region detection > Text
    |   |   |                              ordering > Text recognition > Filtering obtained text
    │   |   ├── prompt_builder.py       <- Code for building system and user prompts for LLM
    │   |   ├── recommendation.py       <- Code for running recommendation pipeline - takes LLM output
    |   |   |                              and complete book data > creates recommendation based on
    |   |   |                              user preference
    │   |   ├── result_merger.py        <- Code for mergeing OCR obtained results and human corrections
    │   |   ├── review_queue.py         <- Code for managing redis during librarian review stage
    │   |   ├── test_parser.py          <- Code for parsing useful text from raw OCR results
    │   |   └── vector_store.py         <- Code for configuring Vector DB and managing its operations
    |   |
    │   └── __init__.py                 <- Makes app a python module
    |   
    │
    ├── frontend                        <- application frontend source code
    │   ├── static                      <- all static resources
    │   |   ├── css
    │   |   |      ├── about.html       <- HTML carrying all information about the application
    │   |   |      └── main.css         <- CSS file for style settings
    │   |   └── js
    │   |          ├── about.js         <- JS for dynamically loading about HTML
    │   |          ├── api.js           <- JS for managing API calls
    │   |          ├── app.js           <- JS for wiring app backend with frontend
    │   |          ├── librarian.js     <- JS for managing librarian workflow
    │   |          ├── reader.js        <- JS for managing reader workflow
    │   |          └── ui.js            <- JS for managing all UIs
    |   |    
    │   └── templates
    │       └── index.html              <- HTML for app index page
    |   
    |   
    ├── docker                          <- All Docker files
    │   ├── celery_worker
    │   |   └── Dockerfile              <- Docker file for building docker image for celery worker 
    │   └── fastapi
    │       └── Dockerfile              <- Docker file for building docker image for fastapi service
    │ 
    │ 
    ├── scripts                         <- Bash scripts for VM deployment and Image push to Registry   
    │   ├── azure-vm-deploy.sh        
    │   └── local-acr-push.sh          
    │ 
    │ 
    ├── tests                           <- All test files
    │   ├── integration                 <- All integration tests
    │   ├── unit                        <- All unit tests
    │   |   └── pipeline                <- Unit tests for various stages of pipelines
    │   |   └── services                <- Unit tests for various stages of all the services
    │   └── conftest.py                 <- Code for setting up test configurations
    │ 
    ├── .env.example                        <- File to load app environment variables on docker run
    ├── docker-compose.dev.yml              <- File to run docker compose stack for app
    ├── pytest.ini                          <- File for setting up pytest configuration
    │ 
    ├── requirements-fastapi-service.txt    <- Requirements file for reproducing environment 
    │                                          conducive for fastapi service; for Docker use
    │
    ├── requirements-worker-service.txt     <- Requirements file for reproducing environment 
    │                                          conducive for celery worker service; for Docker use
    ├── .dockerignore
    ├── .gitattributes
    └── .gitignore


--------