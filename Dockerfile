FROM public.ecr.aws/lambda/python:3.8

RUN pip install torch transformers fastapi uvicorn python-multipart requests Pillow

COPY main.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
