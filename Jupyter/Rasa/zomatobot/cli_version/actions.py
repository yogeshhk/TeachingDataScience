
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from datetime import datetime

import logging
import requests
import json
import os
from rasa_sdk import Action
from rasa.core.events import SlotSet

logger = logging.getLogger(__name__)

# Constants set up for API
ZOMATO_API_KEY = os.environ.get('ZOMATO_API_KEY',"")# Set your Zomato API key in Environment variable
API_URL = 'https://developers.zomato.com/api/v2.1/'
HEADERS = {
'User-agent': 'curl/7.43.0',
'Accept': 'application/json',
'user_key': ZOMATO_API_KEY
}

# curl -X GET --header "Accept: application/json" --header "user_key: MY_API_KEY_HERE" "https://developers.zomato.com/api/v2.1/geocode?lat=41.10867962215988&lon=29.01834726333618"

class ActionRestaurantSearch(Action):
	def name(self):
	# define the name of the action which can then be included in training stories
		return 'action_restaurant_search'

	def parse_search(self, restaurants):
		data = {
			'name' : [],
			'cuisines' : [],
			'address' : [],
			'rating' : [],
			'cost' : [],
		}

		for rest in restaurants:
			rt = rest['restaurant']
			data['name'].append(rt['name'])
			data['cuisines'].append(rt['cuisines'])
			data['address'].append(rt['location']['address'])
			data['rating'].append(rt['user_rating']['aggregate_rating'])
			data['cost'].append(str(rt['average_cost_for_two']))

		return data

	def get_location(self, location):
		# fetch location id
		req_url = API_URL + 'locations?query=' + location

		res = requests.get(req_url, headers=HEADERS)

		# default delhi lat and long
		latitude = 28.625789
		longitude = 77.210276

		if res.status_code == 200:
			latitude = res.json()['location_suggestions'][0]['latitude']
			longitude = res.json()['location_suggestions'][0]['longitude']

		return str(latitude), str(longitude)

	def run(self, dispatcher, tracker, domain):
		# what your action should do
		print("In action server ...")

		location = tracker.get_slot('location') or 'gurgoan'
		cuisine = tracker.get_slot('cuisine') or 'north indian'
		print("Location :{}".format(location))
		print("Cuisine :{}".format(cuisine))

		dispatcher.utter_message(location+cuisine)

		latitude, longitude = self.get_location(location)
		if ZOMATO_API_KEY == "":
			dispatcher.utter_message("Need to define environment variable ZOMATO_API_KEY with key from " + API_URL)
			return []
		req_url = API_URL + 'search?q=' + cuisine + '&lat=' +latitude+ '&lon=' + longitude + '&sort=rating'
		print("Request URL :{}".format(req_url))

		res = requests.get(req_url, headers=HEADERS)

		if res.status_code == 200:
			print("Request successful ...")

			restaurants = self.parse_search(res.json()['restaurants'])
			out_greet_msg = '*Here are top results for {} in {}*'.format(cuisine, location)
			dispatcher.utter_message(out_greet_msg)
			# print(len(restaurants))
			# prepare the output to be sent out
			if len(restaurants) > 0:
				output = []
				# print(restaurants)
				for idx, rest in enumerate(restaurants):
					try:
						out_st = 'Restaurant: ' + str(restaurants['name'][idx]) + "\n" \
										+ 'Cuisines: '+ str(restaurants['cuisines'][idx]) +'\n' \
										+ 'Address: '+ str(restaurants['address'][idx])+'\n' \
										+ 'Rating: '+ str(restaurants['rating'][idx])+'\n' \
										+ 'Average cost for two: '+ str(restaurants['cost'][idx])+'\n'
						output.append(out_st)
						dispatcher.utter_message(out_st)
					except Exception as e:
						print("Problem fetching additional info ...") # length of restaurants is 5 but actual data is of 3, giving out of range error
				output = '\n'.join(output)
			else:
				dispatcher.utter_message('No Restaurant found :( Please try again!')
		else:
			print("Request unsuccessful ...")

			dispatcher.utter_message('FAILED.')
		# dispatcher.utter_message(output)
		return []
