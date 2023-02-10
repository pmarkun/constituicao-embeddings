# Law Analyzer

O Law Analyzer é uma ferramenta de análise de leis. Ele utiliza a API de Embedding da OpenAI para criar embeddings dos documentos (neste caso, seções de leis) e compará-los com a consulta de um usuário. As seções mais relevantes são selecionadas e retornadas ao usuário.

## Instalação

Para instalar o Law Analyzer, você precisa ter o Python3 e o Pip3 instalados em seu sistema. 

Clone o repositório e execute o seguinte comando dentro da pasta do repositório:

```
pip install -r requirements
```

## Uso

Você precisará adquirir uma chave da API da OpenAI e exportá-la como uma variável de ambiente `OPENAI_APIKEY`.

Você também precisará criar um dataframe no formato esperado pela aplicação e salvar como um arquivo csv na pasta `data/` com o nome especificado no construtor da classe LawAnalyzer.

Depois de configurar o ambiente, você pode utilizar o Law Analyzer da seguinte maneira:

```python
from law_analyzer import LawAnalyzer

la = LawAnalyzer(project="example", debug=True)

results = la.similar("O que é a lei da meia-noite?")

print(results)

## Desenvolvimento

Este projeto foi desenvolvido como uma forma de aprendizado. Qualquer contribuição é bem-vinda.

Para contribuir, siga as instruções de Instalação e faça um fork do repositório. Crie uma branch para a sua funcionalidade e abra um pull request com a descrição da sua funcionalidade/correção.