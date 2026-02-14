# Roteiro do Video - LSTM para Predicao de Precos de Acoes

## Informacoes Gerais
- **Duracao estimada:** 10 a 14 minutos
- **Plataforma:** YouTube (educacional, publico)
- **Tom:** Didatico, tecnico mas acessivel
- **Titulo sugerido:** "Prevendo Precos de Acoes com LSTM em PyTorch -- Do Zero ao Deploy"

---

## ABERTURA (0:00 - 1:00)

**[Tela: slide de titulo ou editor de codigo]**

"Ola, pessoal! Meu nome e Guilherme Favaron e neste video vou mostrar como construir um projeto completo de predicao de precos de acoes usando redes neurais LSTM em PyTorch -- do zero ate o deploy em producao."

"Vamos passar por cada etapa: coleta de dados, pre-processamento, construcao e treinamento do modelo, avaliacao de metricas, e por fim colocar tudo online com uma API e um dashboard interativo."

"Escolhi trabalhar com a PETR4, acao da Petrobras, mas a mesma abordagem funciona para qualquer ticker. Se voce quer aprender Deep Learning aplicado a series temporais financeiras de forma pratica, fica ate o final."

---

## PARTE 1: POR QUE LSTM PARA ACOES? (1:00 - 2:00)

**[Tela: slides ou diagrama explicativo]**

"Antes de entrar no codigo, por que usar LSTM para prever precos?"

"Precos de acoes sao series temporais -- o valor de hoje depende do que aconteceu nos dias anteriores. Redes neurais tradicionais nao capturam bem essa dependencia temporal. Ja a LSTM, que e um tipo de rede recorrente, foi projetada exatamente para isso: ela tem uma memoria interna que permite aprender padroes de longo prazo nos dados."

"O que vamos construir: um modelo que recebe os ultimos 60 dias de precos de fechamento e preve o preco do proximo dia. Simples de entender, mas surpreendentemente eficaz."

"Importante: no final do video vou falar sobre as limitacoes reais desse tipo de modelo. Previsao de acoes e um problema muito mais complexo do que so olhar historico de precos."

---

## PARTE 2: NOTEBOOK - DADOS E MODELO (2:00 - 5:30)

**[Tela: Google Colab com o notebook aberto]**

### Coleta de Dados (2:00 - 2:45)
"Comecei usando a biblioteca yfinance para baixar o historico da PETR4 desde 2018. Sao mais de mil dias de negociacao."

*[Executar a celula de download e mostrar o DataFrame]*

"Cada linha e um dia de negociacao com preco de abertura, maxima, minima, fechamento e volume. Para a LSTM, vamos usar apenas o preco de fechamento."

### Analise Exploratoria (2:45 - 3:30)
"Na analise exploratoria, criei graficos interativos com Plotly mostrando a evolucao do preco ao longo do tempo, medias moveis de 50 e 200 dias, e a distribuicao dos retornos diarios."

*[Mostrar os graficos gerados]*

"Aqui da pra ver claramente os ciclos da Petrobras, a volatilidade em periodos de crise, e como o preco se comporta em relacao as medias moveis. Essa visualizacao ja ajuda a entender se existe um padrao que o modelo pode aprender."

### Pre-processamento (3:30 - 4:15)
"O pre-processamento tem tres etapas essenciais que fazem toda a diferenca:"

"Primeiro, normalizacao com MinMaxScaler -- a LSTM funciona muito melhor com dados entre 0 e 1. Esse e um passo que muita gente esquece e depois nao entende por que o modelo nao converge."

"Segundo, criacao de janelas deslizantes -- o modelo recebe os 60 dias anteriores para prever o proximo. Esse numero 60 e um hiperparametro que voce pode ajustar."

"Terceiro, e talvez o mais importante: split temporal. Dividimos 70% treino, 15% validacao, 15% teste. Mas sem embaralhar! Em series temporais, se voce embaralhar os dados, esta vazando informacao do futuro para o passado, e o modelo vai parecer muito melhor do que realmente e."

*[Mostrar o codigo e os prints do shape dos dados]*

### Modelo e Treinamento (4:15 - 5:30)
"O modelo e uma LSTM com duas camadas de 128 neuronios cada, seguida de camadas fully connected que geram a previsao final. Adicionei dropout de 0.2 entre as camadas para evitar overfitting."

*[Mostrar a classe LSTMModel]*

"Para o treinamento, usei Adam como otimizador, MSE como funcao de perda, learning rate scheduler para ir reduzindo o learning rate quando o modelo para de melhorar, e early stopping com paciencia de 15 epocas."

*[Mostrar a curva de treinamento -- loss caindo]*

"Aqui no grafico vemos a loss caindo tanto no treino quanto na validacao. Quando as duas curvas acompanham, e um bom sinal -- significa que o modelo esta aprendendo padroes reais e nao memorizando os dados."

---

## PARTE 3: AVALIACAO (5:30 - 6:30)

**[Tela: graficos de avaliacao no notebook]**

"As metricas no conjunto de teste foram:"
*[Mostrar os valores de MAE, RMSE e MAPE]*

"Para quem nao esta familiarizado: o MAE e o erro medio em reais -- ou seja, em media o modelo erra X reais. O RMSE penaliza mais os erros grandes. E o MAPE e o erro percentual, que facilita a interpretacao."

"O grafico mais revelador e esse: valores reais em azul versus previsoes em laranja. Notem como o modelo acompanha bem a tendencia geral."

*[Mostrar o grafico Real vs Previsto]*

"E aqui a visao completa com treino, validacao e teste, mostrando que o modelo generaliza bem para dados que nunca viu. Isso e fundamental -- qualquer modelo pode decorar dados de treino."

---

## PARTE 4: API E DASHBOARD (6:30 - 9:00)

**[Tela: navegador com o HuggingFace Spaces aberto]**

### Dashboard Gradio (6:30 - 7:30)
"Agora a parte que eu acho mais legal: colocar o modelo em producao. Deployei tudo no HuggingFace Spaces usando Docker."

*[Navegar pelo dashboard]*

"O dashboard tem tres abas. Na primeira, voce digita qualquer ticker brasileiro, clica em gerar previsao, e o sistema baixa os dados mais recentes, roda o modelo e mostra o grafico com a previsao para o proximo dia."

*[Clicar em 'Gerar Previsao' e mostrar o resultado]*

"A segunda aba mostra informacoes detalhadas sobre o modelo: arquitetura, metricas, hiperparametros. A terceira documenta os endpoints da API para quem quiser integrar."

### API REST (7:30 - 8:30)
"Alem do dashboard visual, tem uma API REST completa rodando por baixo com FastAPI. Isso significa que qualquer sistema pode consumir as previsoes programaticamente."

*[Abrir /docs no navegador]*

"Aqui no Swagger temos os endpoints. O GET /health verifica se a API esta no ar. O GET /predict/PETR4.SA faz a previsao automaticamente."

*[Executar o endpoint /predict no Swagger e mostrar a resposta JSON]*

"E o GET /metrics retorna tanto as metricas do modelo quanto metricas operacionais como latencia e numero de requisicoes. Isso e essencial para monitoramento em producao."

### Modelo no HuggingFace Hub (8:30 - 9:00)
"O modelo treinado tambem esta publicado no HuggingFace Hub, entao qualquer pessoa pode baixar e usar nos seus proprios projetos."

*[Mostrar a pagina do modelo no HF Hub]*

"Tem o model card documentando tudo: arquitetura, como usar, limitacoes, e os arquivos para download."

---

## PARTE 5: API REST E TESTES (9:00 - 10:00)

**[Tela: terminal com curl]**

"Alem do dashboard visual, a API REST permite que qualquer sistema consuma as previsoes programaticamente. Vou mostrar como testar."

*[Executar no terminal]*

"Para verificar se a API esta no ar:"
```
curl https://guifav-lstm-petr4-stock-prediction.hf.space/health
```

"Para obter a previsao do proximo dia:"
```
curl https://guifav-lstm-petr4-stock-prediction.hf.space/predict/PETR4.SA
```

"E para ver metricas do modelo e da API:"
```
curl https://guifav-lstm-petr4-stock-prediction.hf.space/metrics
```

*[Mostrar as respostas JSON]*

"Tudo documentado com Swagger. Basta acessar /docs no navegador."

---

## PARTE 6: INFRAESTRUTURA E RETREINO (10:00 - 11:00)

**[Tela: Dockerfile e GitHub Actions no editor]**

"Sobre a infraestrutura: o projeto inteiro roda em um container Docker. Isso garante que funciona em qualquer maquina sem problemas de dependencia."

*[Mostrar o Dockerfile]*

"Para rodar localmente, basta um docker-compose up. O healthcheck monitora a API automaticamente."

"O monitoramento e feito com logging estruturado -- cada requisicao registra latencia, endpoint e status. Esses dados ficam acessiveis via endpoint /metrics."

*[Mostrar o arquivo .github/workflows/retrain.yml]*

"E aqui um detalhe que faz toda a diferenca em producao: configurei um GitHub Action que toda segunda-feira, antes da abertura do mercado, retreina o modelo automaticamente com os dados mais recentes. O workflow baixa os dados atualizados, retreina o modelo, publica os novos artefatos no HuggingFace Hub e atualiza o Space. Tudo automatico, sem intervencao manual."

---

## ENCERRAMENTO (11:00 - 12:00)

**[Tela: slide de resumo ou repositorio no GitHub]**

"Recapitulando o que construimos: coletamos dados financeiros, treinamos uma LSTM em PyTorch, avaliamos com metricas de regressao, deployamos com FastAPI e Gradio, containerizamos com Docker, publicamos tudo no HuggingFace, e configuramos retreino automatico semanal com GitHub Actions."

"Todo o codigo esta no GitHub, o notebook roda direto no Google Colab sem instalar nada, o modelo esta no HuggingFace Hub, e o dashboard esta live. Os links estao na descricao."

"Agora, o ponto mais importante do video: esse modelo e educacional. Precos de acoes sao influenciados por politica, macroeconomia, sentimento de mercado, noticias -- fatores que um modelo baseado so em historico de precos nao captura. Use esse projeto para aprender, nao para operar na bolsa."

"Se quiser evoluir esse projeto, aqui vao algumas ideias: adicionar mais features como volume e indicadores tecnicos, testar arquiteturas como Transformer ou modelos hibridos CNN-LSTM, implementar previsao multi-step, ou adicionar backtesting."

"Espero que esse video tenha sido util pra voce. Se curtiu, deixa o like, se inscreve no canal, e qualquer duvida comenta aqui embaixo. Valeu!"

---

## DESCRICAO DO VIDEO (para copiar no YouTube)

```
Projeto completo de predicao de precos de acoes usando redes neurais LSTM em PyTorch -- do zero ao deploy em producao.

Neste video mostro cada etapa do pipeline:
- Coleta de dados financeiros com yfinance (PETR4.SA - Petrobras)
- Pre-processamento e criacao de janelas temporais
- Construcao de modelo LSTM com 2 camadas em PyTorch
- Treinamento com early stopping e learning rate scheduling
- Avaliacao com MAE, RMSE e MAPE
- Deploy com FastAPI + Gradio no HuggingFace Spaces
- API REST com endpoints testados via curl
- Publicacao do modelo no HuggingFace Hub
- Containerizacao com Docker
- Retreino automatico semanal com GitHub Actions

Links:
- Dashboard live: https://huggingface.co/spaces/guifav/lstm-petr4-stock-prediction
- Modelo: https://huggingface.co/guifav/lstm-petr4-stock-prediction
- Repositorio: https://github.com/guifav/previsao_preco_acoes
- Notebook: [link do Colab]

Tags: LSTM, PyTorch, Machine Learning, Deep Learning, Stock Prediction,
FastAPI, HuggingFace, Docker, Series Temporais, Python Tutorial
```

## TAGS SUGERIDAS

LSTM, PyTorch, machine learning, deep learning, predicao acoes, stock prediction, FastAPI, HuggingFace, Docker, series temporais, Petrobras, PETR4, redes neurais, Python, tutorial, deploy, API, Gradio, GitHub Actions, CI/CD, retreino automatico
