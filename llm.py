#!/usr/bin/env python3

import sys, time
import warnings
import logging
from colorama import Fore, Back, Style, init

init(autoreset=True)
BOLD = '\033[1m'
warnings.filterwarnings("ignore")

class LLM:
	def __init__(self):
		bnrSty=Back.RED + Fore.BLACK + BOLD
		print(bnrSty+" *------------------------------------------------------------------------------* ")
		print(bnrSty+" |                LLM Chat+RAG [ llamaIndex,langGraph, Milvus ]                 | ")
		print(bnrSty+" |                             by : Satyam Swarnkar                             | ")
		print(bnrSty+" *------------------------------------------------------------------------------* ")

if __name__ == "__main__":
	app = LLM()