# Projeto House Sales in King County, USA

![dataset-cover](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/06e4180a-7475-4638-88e0-47b17c797162)

Após realizar análises, explorei diversas hipóteses que resultaram em insights intrigantes no âmbito do projeto "House Sales in King County, USA" de Análise e Ciência de Dados do Kaggle, voltado para a previsão de preços de imóveis.

Esse projeto fictício visa conduzir uma análise exploratória de dados em busca do modelo ideal de previsão de preços para residências no Condado de King, utilizando regressões simples. Trata-se de uma proposta de competição do Kaggle, e o conjunto de dados necessário está disponível em Kaggle, sob o título "House Sales in King County, USA".

# 1. Questões de Negócio
   
A operação da empresa é baseada no seguinte modelo de negócio: adquirir propriedades a valores mais baixos e posteriormente comercializá-las por um preço superior. Os dados disponíveis abrangem os valores de venda de residências para o Condado de King, englobando Seattle, durante o período compreendido entre maio de 2014 e maio de 2015.

### 1.1. Problemática
A equipe de negócios da empresa enfrenta desafios ao identificar as oportunidades mais vantajosas para aquisição de imóveis. A principal dificuldade reportada por eles está relacionada à vasta quantidade de dados e à necessidade de filtrá-los para identificar as propriedades mais adequadas para compra. Com o intuito de obter uma compreensão mais aprofundada do contexto empresarial, busco abordar algumas perguntas comuns neste setor.

### 1.2. Objetivo
Sugerir as propriedades mais vantajosas para aquisição, incluindo uma recomendação de margem de lucro.

# 2. Premissas de negócio
   
Os atributos do conjunto de dados são:

| Atributos  | Descrição |
| --- | --- |
| id | Identificador  |
| date | Data que o imóvel foi vendido |
| bedrooms | Quantidade de quartos |
| bathrooms | Quantidade de banheiros |
| sqft_living  | Pés quadrados construídos |
| sqft_lot | Pés quadrados do terreno |
| floors | Quantidade de andares |
| waterfront | Se é de frente para a água: Sim (1) ou Não (0) |
| view  | Índice de 0 – 4 com relação a vista do imóvel |
| condition  | Índice de 1 – 5 com relação as condições que se encontram a casa |
| grade | Nota de 1 – 13 com relação a qualidade dos materiais e mão de obra utilizada na construção |
| sqft_above  | Pés quadrados construídos sem considerar o porão |
| sqft_basement  | Pés quadrados do porão |
| yr_built  | Ano que começou a construção |
| yr_renovated  | no da última reforma. Zero (0) indica que não foi reformada |
| zipcode  | Código ZIP do imóvel |
| lat | Latitude  |
| long | Longitude |
| sqft_living15 | Média dos pés quadrados construídos dos 15 imóveis mais próximos |
| sqft_lot15 | Média dos pés quadrados do terreno dos 15 imóveis mais próximos |

- Não abordaremos minuciosamente as quantidades não inteiras de quartos, andares e banheiros.
- Consideraremos que os imóveis classificados com os valores 1 e 2 no atributo 'conditions' estão em condições precárias, indicando a necessidade de reformas.

**Para a limpeza dos dados:**
Ao examinar todas as características do conjunto de dados, juntamente com uma tabela de correlação e alguns gráficos, cheguei à conclusão de que as variáveis mais impactantes no aumento do preço incluem: área interna, avaliação da construção, vista da casa, condição, quantidade de banheiros, quantidade de quartos e a presença de vista para o mar.

Optei por empregar a mediana devido à considerável variação nos preços, o que resultava em valores muito acima ou abaixo do padrão no conjunto de dados, gerando resultados distorcidos.

# 3. Planejamento da solução
### 3.1. Produto final
- Entregar Insights dos dados;
- Fazer recomendações de compras de imóveis;
- Modelar o algoritmo e Converter o modelo Para Resultados Financeiros do Negócio.

### 3.2. Ferramentas
- Google Collab
- Visual Studio Code
- Github
- Estatística para Análise de Dados (média, mediana, desvio padrão)
- Modelo de Machine Learning

# 4. Insights dos dados
Conduzindo análises aprofundadas, explorei diversas conjecturas que resultaram em insights interessantes.

Os gráficos revelam que, apesar de existir uma tendência na relação entre preço e características, onde este aumenta conforme o crescimento da variável (tamanho, vista, etc.), observamos que os valores não seguem uma trajetória linear, sugerindo a presença possivelmente de outros fatores que influenciam no aumento do preço em algumas residências.

Nesse cenário, é relevante destacar a importância das regiões onde as casas estão situadas, pois esse aspecto pode representar o principal impulsionador da valorização, indo além das características apresentadas. Com esse propósito, elaborei um mapa gráfico para verificar a existência dessa tendência, categorizando as regiões em níveis de 1 a 5 de acordo com o boxplot dos preços.

![MÉDIA DOS PREÇOS ÁREA INTERNA](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/3edadabc-cdf0-4822-b8fd-3b6518356f1f)

![MÉDIA DOS PREÇOS NÍVEL DA VISTA](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/f7ddda27-cf11-44e2-b3b3-b5718b9174b1)


![MÉDIA DOS PREÇOS NÚMERO DE QUARTOS](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/0dab8990-6193-4547-a7f0-71bc7f3bc1d0)

![M´DIA DOS PREÇOS CONDIÇÕES](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/a147c2df-82ac-442e-bfba-236b2b6a5651)

![MÉDIA DOS PREÇOS SAZONALIDADE](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/228da43f-7a22-4358-a3fe-056242bb42d5)

![MÉDIA DOS PREÇOS VISTA MAR](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/c440abdc-eaf3-4f14-8b36-cda4d553fbb2)

**H1:** Dividi os dados em cinco níveis distintos:

Nível 1: Valores menores que $323.000
Nível 2: Valores entre $323.000 e $450.000
Nível 3: Valores entre $450.000 e $645.000
Nível 4: Valores entre $645.000 e $1.127.500
Nível 5: Valores acima de $1.127.500

![MAPA ESCALA COM NIVÉIS](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/b18fcdfd-1a52-4d85-96f9-15492599d376)

Como pode ser observado, há uma concentração significativa na região central, especialmente em Bellevue, Seattle e Lake Sammamish. No entanto, apesar dessa concentração, é evidente que há casas de alto valor em outras áreas. Para uma segmentação mais precisa dos dados, optei pela divisão por CEP (zipcode), revelando uma diferença expressiva nos preços entre as diferentes áreas. O CEP de maior valor atinge $1.892.500, enquanto o de menor valor é de $235.000.

Essa abordagem revelou uma correlação mais notável quando o conjunto de dados foi dividido por CEP, destacando a importância de analisar cada área separadamente. Ao separar as casas em categorias de baixo e alto valor, considerando a mediana para cada CEP, tornou-se mais evidente a influência de fatores como a quantidade de banheiros ou quartos, por exemplo.

**H2:** Classifiquei os dados em cinco níveis distintos:
Nível 1: Áreas menores que 1.430 p²
Nível 2: Áreas entre 1.430 p² e 1.920 p²
Nível 3: Áreas maiores que 1.920 p² e menores ou iguais a 2.550 p²
Nível 4: Áreas maiores que 2.550 p² e menores ou iguais a 4.230 p²
Nível 5: Áreas acima de 4.230 p²

![NÍVEL DA VISTA](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/dbf8f8ee-bed0-4081-b0da-72021a0f7590)

![NÚMERO DE BANHEIROS](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/6539de7c-64be-4758-bd3f-d63fd25f72e7)

![NÚMERO DE QUARTOS](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/79947853-6bd8-4bf1-9683-9bf014544a14)

![TAMANHOS INTERNO](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/b7c4119f-3cde-482d-963c-98305bf12c57)

Essas categorias proporcionaram conclusões notáveis. A mais evidente é a necessidade de que as variáveis estejam alinhadas para justificar um preço elevado. Em outras palavras, não adianta ter 8 quartos se a casa possui apenas 1 banheiro ou uma avaliação baixa. Outro ponto de destaque refere-se à influência da região na precificação, indicando que residências com características semelhantes podem apresentar preços distintos devido à sua localização.

![VARIAÇÃO POR MÊS](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/7c328ed5-d4cf-4619-98bc-65c0545560f4)

Quanto à sazonalidade, ao analisar a distribuição média de preços das casas ao longo dos meses, constatei uma valorização mais expressiva de março a julho, correspondendo ao período de primavera-verão. No entanto, essa diferença é sutil, indicando alterações discretas nos preços ao longo do ano.


# 5. Machine Learning

![modelo 1](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/633feb70-57fa-4ca7-b22e-48ae1a4c81de)

Como podemos notar, alguns modelos apresentam uma pontuação de treinamento significativamente elevada, como é o caso da Árvore de Decisão, que registrou 0.99 no treinamento, mas um desempenho mais modesto nos testes, marcando 0.62. Apesar do score de teste se aproximar de 80% (0.80), ainda não atinge o ideal para previsões de preços, especialmente quando observamos o erro médio absoluto (mean absolute error), que apresentou $75 mil como a menor variância.

Diante desses resultados, decidi realizar testes de validação cruzada com apenas três dos modelos: Random Forest Regressor, SVR e Rede Neural. Utilizando o GridSearch, identifiquei os melhores parâmetros para cada um dos modelos. Com os resultados em mãos, conduzi a validação cruzada dividindo meu conjunto de dados em 10 partes (n_splits), alternando entre partes de teste e treinamento para obter 10 resultados e calcular a média. Repeti esse processo 30 vezes para cada modelo, totalizando 300 testes. O objetivo é criar uma tabela com 30 resultados, sendo que cada resultado é a média da validação cruzada com 10 resultados. 


Como etapa decisiva na escolha do modelo, realizei o teste de ANOVA e Tukey para verificar se há uma diferença estatisticamente significativa entre os modelos ou se as discrepâncias entre eles podem ser consideradas insignificantes.

![modelo2](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/9eefd150-8bc4-4d0b-97a4-0cac52d80f63)

![modelo3](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/aa56ebfe-63c6-4eb1-9cd6-3bfd13d7c6fb)

Os resultados indicam que, de fato, os algoritmos apresentam diferenças significativas entre si. Ao considerar a média das pontuações de cada modelo, a rede neural emerge como a escolha mais acertada para o conjunto de dados avaliado.



# 6. Resultados
### 6.1. Resultados financeiros
   
Quais casas deveriam ser compradas e por qual valor?

A resposta a essa pergunta pode ser encontrada ao empregar o modelo para determinar o preço justo dos imóveis com base no algoritmo. Em outras palavras, o modelo indicará o valor ideal a ser cobrado por cada propriedade. Dessa maneira, ao comparar as previsões de preço com os valores reais, é possível identificar discrepâncias. Se a quantia prevista for superior, significa que a casa está subvalorizada; se for inferior, a casa está sobrevalorizada.
Um exemplo prático de como o modelo pode ser utilizado envolve a criação de um mapa que permite ao CEO visualizar as casas que estão abaixo e acima do preço, exibindo tanto o valor real quanto o preço justo determinado pelo algoritmo.

**Exemplo:**
Valor real: $305.000
Valor justo: $333.341
Lucro: $28.341

Em um cenário em que a empresa adquira todas as casas abaixo da previsão e as venda pelo valor justo, o potencial de lucro estimado seria aproximadamente $761.564.426. No entanto, é importante ressaltar que esse valor é uma estimativa, pois deve-se levar em consideração a margem de erro inerente ao algoritmo.


Uma vez a casa em posse da empresa, qual o melhor momento para vendê-la e qual seria o preço da venda?

Conforme evidenciado na análise, observa-se uma inclinação ascendente nos preços durante o período de primavera-verão na região de King County. Estatisticamente, embora não seja um aumento extraordinário, é uma tendência palpável.
Ao considerar o preço médio das casas vendidas em ambas as hipóteses (primavera-verão e outono-inverno), verifica-se um acréscimo de cerca de 6%.

![SAZONAL](https://github.com/mariacdev/Projeto-House-Sales-in-King-County-USA-/assets/134116444/20acd5a2-034e-4071-856b-5da570747a77)

Satisfazendo todos esses critérios, a estratégia mais vantajosa consiste em realizar aquisições durante o outono-inverno e efetuar as vendas durante a primavera-verão. A obtenção de dados adicionais referentes a pelo menos mais dois anos, como as vendas de 2012 e 2013, permitiria a verificação de um padrão em relação ao mês de pico de vendas.

# 7. Conclusões

Após uma minuciosa análise dos imóveis, foi possível identificar propriedades com qualidade superior em comparação com aquelas mais próximas, e com preços significativamente abaixo do padrão de mercado.

Observa-se que, embora existam variáveis com uma correlação positiva com o preço, elas não são suficientes para criar um modelo altamente eficaz. O conjunto de dados apresenta diversos "outliers," cujas influências podem ser atribuídas a fatores externos ao dataset. Por exemplo, a localização da casa em um quarteirão com um centro comercial, uma obra pública, um parque ou qualquer elemento que agregue valor em termos de entretenimento, turismo, facilidades de serviços ou acesso a outras regiões. Além disso, acordos específicos de venda, como descontos, também podem explicar a presença desses "outliers."

Além disso, os insights fornecidos à equipe de negócios podem ser uma valiosa contribuição para a identificação de outras oportunidades imobiliárias, além de abrir caminho para um novo modelo de negócios para a empresa.

# 8. Próximos passos

Conforme mencionado anteriormente, as análises realizadas proporcionaram insights que abrem caminho para duas oportunidades significativas de negócio:

- Explorar as potencialidades de expansão da empresa através de um novo modelo de negócio: adquirir imóveis em condições ruins, reformá-los e revendê-los.

- Avaliar a qualidade do algoritmo. Planejo retornar a esse modelo quando possuir um conhecimento mais sólido, caso perceba oportunidades de aprimoramento.











