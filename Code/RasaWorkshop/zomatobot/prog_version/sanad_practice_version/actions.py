# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import requests
import json

from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet

logger = logging.getLogger(__name__)

class ActionRestaurantSearch(Action):
	def name(self):
	# define the name of the action which can then be included in training stories
		return 'action_restaurant_search'

	def run(self, dispatcher, tracker, domain):
		# what your action should do
		return []
