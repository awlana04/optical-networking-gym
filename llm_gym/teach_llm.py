import os
import re

from langchain_ollama import ChatOllama
from langchain_core import SystemMessage, HumanMessage

class TeachLLM:
  '''
    Teaches the LLM to choose the best heuristic, first is sent by a System Message to speculate how would be the action, service blocking rate and mean GSNR in 4 different heuristics by just providing the network source and path.

    To run it, just call the instance run().

    Args:
      source: Network source.
      path: Network path.
      service_blocking_rate: Network info provided during episodes.
      mean_gsnr: Network info provided during episodes.
  '''

  global response

  global actions
  global choosen_heuristic
  global returned_service_blocking_rate
  global returned_mean_gsnr

  def __init__ (self, source, path, service_blocking_rate, mean_gsnr):
    self.source = source
    self.path = path
    self.service_blocking_rate = service_blocking_rate
    self.mean_gsnr = mean_gsnr

  def callLLM (self, messages):
    '''Invokes the endpoint to an LLM'''

    chat = ChatOllama(model="qwen3-coder-next:q4_K_M", base_url=os.getenv("LLM_BASE_URL"))

    response = chat.invoke(messages)

    print (response.content)
  
  def determineHeuristic (self):
    '''Determines best heuristic based on source and path according to the action higher value received'''
    messages = [
      SystemMessage(content="Considere um source como a fonte de uma rede atual e um path como sendo o caminho para a rede atual. Você é capaz de escolher um número de 1 à 4 que irá representar 4 heurísticas diferentes: 1. First Fit, 2. Lowest Spectrum, 3. Load Balancing Modulation, 4. Load Balancing Best Modulation. As heurísticas irão receber o source e o path e irão retornar uma action. Você deve escolher àquela que tiver o menor Service Blocking Rate e Mean GSNR. A heurística escolhida irá determinar a melhor estratégia de comunicação de uma fibra óptica elástica a partir do estado da rede atual, o source e o path. Justifique a sua escolha. Responda explicitando a heurística escolhida e qual foi o valor que a action atingiria com a heurística escolhida em relação as outras, assim como os direfentes Service Blocking Rate e Mean GSNR, por exemplo, Heurística: 1, Action 1: 500, Service Blocking Rate 1: 0.18181818, Mean GSNR 1: 20.47558723, Action 2: 200, Service Blocking Rate 1: 0.139888810, Mean GSNR 1: 19.98476174499."),

      HumanMessage(content=f"Retorne de 1 à 4 a melhor heurística para o cenário do seguinte path: {self.path} e para o seguinte source: {self.source}.")
    ]

    self.callLLM(messages)

  def evaluateLLM (self):
    '''Evaluates the heuristic choosen by comparing the actions, service blocking rate and mean_gsnr received'''

    # Separates the action results in an array
    actions = re.findall(response.content, r"Action\s*\d*:\s*(\d+)") 
    # Takes the choosen heuristic number
    choosen_heuristic = re.findall(response.content, r"Heurística: \d+")[0]

    returned_service_blocking_rate = re.findall(response.content, r"Service Blocking Rate \s*\d*: \d+").append(self.service_blocking_rate)
    returned_mean_gsnr = re.findall(response.content, r"Mean GSNR \s*\d*: \d+").append(self.mean_gsnr)

    # This is a ranking to the best canditates to heuristic, the index represents the heuristic
    ranking = [0,0,0,0]

    # We supposse the graph_load test works only with the first fit heuristic, because of this, we verify if the LLM has choosen a different one, if it did, we ask him for to choose between our first fit heuristic and the choosen one. Also asks it to look for another canditade.
    if (choosen_heuristic != 1):
      messages = [
        SystemMessage(content="Você é capaz de determinar a melhor a heurística baseada na comparação entre a action atual, service blocking rate, mean GSNR de uma rede óptica elástica, lhe será fornecido 2 heurísticas distintas e você deve: 1. A partir dos dados fornecidos, fazer uma comparação entre os menores service blocking rate e mean GSNR, 2. A maior action também contará pontos, 3. Considere que a heurística pode ser, a. First Fit, b. Lowest Spectrum, c. Load Balancing Modulation e, d. Load Balancing Best Modulation, 4. Cada action representa uma heurística diferente assim como as posições do service blocking rate e do mean GSNR, por exemplo, se lhe fornecerem uma heurística de valor 2, representa a action 2, service blocking rate 2 e mean GSNR 2. 5. Você deve analisar se uma outra heurística além das fornecidas pode ser melhor, por exemplo, lhe forneceram as heurística 1 e 2, mas a partir de suas análises, você descobre que a 3 é uma canditada próxima, então você deverá julgar entre a 1, 2 e 3. Assim, você deve escolher a melhor heurística, justifique a sua escolha, caso você perceba um possível canditado, justifique porque há mais um canditado. Desconsidere um novo canditado para a Action 1."),
        HumanMessage(content=f"Julgue a melhor heurística baseada nas 4 actions: {actions}, nos Service Blocking rate: {returned_service_blocking_rate} e o Mean GSNR: {returned_mean_gsnr}, você tem que escolher entre as heurísticas: 1 e {choosen_heuristic}")
      ]

      self.callLLM(messages)

      # Update the choosen heuristics array
      choosen_heuristic = re.findall(response.content, r"Heurística: \d+")

    # Creates the ranking
    for index, action in action:
      if (returned_service_blocking_rate[index] > returned_service_blocking_rate[choosen_heuristic - 1]):
        ranking[index] += 1
      if (returned_mean_gsnr[index] > returned_mean_gsnr[choosen_heuristic - 1]):
        ranking[index] += 1     
      if (action[index] > action[choosen_heuristic - 1]):
        ranking[index] += 1
    
    # Checks if the choosen heuristics is the hightest value in the ranking, if is not, then we return a negative feedback to the LLM
    if (ranking.index(max(ranking)) != choosen_heuristic - 1):
      self.callLLM(HumanMessage(content=f"A heurística: {choosen_heuristic} não possui os melhores parâmetros em relação à heurística: {ranking.index(max(ranking))}."))

  def run(self):
    self.determineHeuristic()
    self.evaluateLLM()

  # def evaluateLLM (self):
  #   '''Evaluates the heuristic choosen by comparing the actions received'''

  #   # Separates the action results in an array
  #   actions = re.findall(response.content, r"Action\s*\d*:\s*(\d+)") 
  #   # Takes the choosen heuristic number
  #   choosen_heuristic = re.findall(response.content, r"Heurística: \d+")[0]

  #   for index, action in actions:
  #     if (action > actions[choosen_heuristic - 1]):
  #       self.callLLM(HumanMessage(content=f"A action: {index + 1} tem o melhor resultado, logo esta deveria ser a heurística escolhida."))
  #     elif (action == actions[choosen_heuristic - 1]):
  #       return
  #     else:
  #       self.callLLM(HumanMessage(content=f"A heurística: {choosen_heuristic} escolhida por você demonstrar está correto de acordo com a action: {actions[choosen_heuristic - 1]}."))

  # def calculateLLMAccuracy (self):
    '''
    Calculates weather the LLM had well returned each action result or not.

    Compares with the results taken during the grap_load test.
    '''