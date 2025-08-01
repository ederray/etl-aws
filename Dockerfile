# Imagem Python 3.11 para aws lambda
FROM public.ecr.aws/lambda/python:3.11

# Definição do diretório de trabalho
WORKDIR /var/task

# Copia os arquivos do seu projeto para o diretório de trabalho
COPY . .

# Adição do diretorio /var/task ao PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/var/task"

# Atualização do pip e instalação das dependências
RUN pip install --upgrade pip
RUN pip install -r src/lambda_cotacao_diaria/requirements.txt

# Definção do handler da lambda (formato: arquivo.função)
CMD ["src.lambda_cotacao_diaria.lambda_function.lambda_handler"]


