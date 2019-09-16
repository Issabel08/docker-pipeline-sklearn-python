FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install  -r requirements.txt

VOLUME /usr/src/app/data_train/ /usr/src/app/data_test/  /usr/src/app/results/

COPY . .

CMD [ "python", "./pipe.py" ]
