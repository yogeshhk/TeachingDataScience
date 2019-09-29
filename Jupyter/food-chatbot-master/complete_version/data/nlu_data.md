<!--- Make sure to update this training data file with more training examples from https://forum.rasa.com/t/rasa-starter-pack/704 --> 

## intent:goodbye  
- Bye 			
- Goodbye
- See you later
- Bye bot
- Goodbye friend
- bye
- bye for now
- catch you later
- gotta go
- See you
- goodnight
- have a nice day
- i'm off
- see you later alligator
- we'll speak soon
- end
- finish

## intent:greet
- Hi
- Hey
- Hi bot
- Hey bot
- Hello
- Good morning
- hi again
- hi folks
- hi Mister
- hi pal!
- hi there
- greetings
- hello everybody
- hello is anybody there
- hello robot
- who are you?
- what are you?
- what's up
- how do you do?

## intent:thanks
- Thanks
- Thank you
- Thank you so much
- Thanks bot
- Thanks for that
- cheers
- cheers bro
- ok thanks!
- perfect thank you
- thanks a bunch for everything
- thanks for the help
- thanks a lot
- amazing, thanks
- cool, thanks
- cool thank you
- yes, thanks!

## intent:affirm
- y
- Y
- yes
- yes sure
- absolutely
- for sure
- yes yes yes
- definitely
- yes, it did.

## intent:restaurant_search
- i'm looking for a place to eat
- I want to grab lunch
- I am searching for a dinner spot
- i'm looking for a place in the [north](location) of town
- show me [chinese](cuisine) restaurants
- show me a [mexican](cuisine) place in the [centre](location)
- i am looking for an [indian](cuisine) spot
- search for restaurants
- anywhere in the [west](location)
- anywhere near [18328](location)
- I am looking for [asian fusion](cuisine) food
- I am looking a restaurant in [29432](location)
- I am hungry
- Can you tell me a place to eat?
- I need food
- Food near [delhi](location)
- Food near me
- what are some good restraunts in [paris](location)
- suggest me some good [chinese](cuisine) place
- suggest me a good place to eat
- where can i get best [mughlai](cuisine)
- where can I find best [burger](cuisine) in [tokyo](location)?
- best [biryani](cuisine) in [new york](location)?
- i want to eat [pasta](cuisine)
- best [burgers](cuisine) in [delhi](location)
- what are the best [pizza](cuisine) in [delhi](location)
- i feel like having [sushi](cuisine)
- i would like to have some [biryani](cuisine)
- best [pizza](cuisine) near me
- i am feeling hungry
- tell me the nearest [burger](cuisine) place
- i want a [sandwich](cuisine)
- where can i get [pizza](cuisine) in [chennai](location)
- can you recommend some good [continental](cuisine) place in [mysore](location)
- Actually, I want to eat [pasta](cuisine) in [chennai](location)
- can you suggest good [biryani](cuisine) place in [chennai](location)?
- [dosa](cuisine) in [gurugram](location)
- [dosa](cuisine) in [gurgaon](location)

## intent:deny
- no
- never
- I don't think so
- don't like that
- no way
- not really
- n
- N

## lookup:location
data/locations.txt

## lookup:cuisine
data/cuisines.txt