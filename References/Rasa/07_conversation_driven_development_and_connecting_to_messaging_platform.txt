# 07 - Messaging Channels, Conversation-Driven Development and Rasa X

## Agenda
1. Conversation-Driven Development
    * Combining user input and software engineering best practices at every stage of the development cycle
    * Conversations = training data
2. Rasa X
    * A tool for CDD
    * Launching Rasa X locally
    * Reviewing Channels
3. Messaging Channels
    * Live-coding : connecting to Telegram
4. Review

## Conversation-Driven Development

CDD captures the lessons learned as a community

If you've built conversational AI before, you know that
* it is a hard problem to solve
* building a prototype is not the hard part
* hard parts all show up when we want to go from a prototype to something we want to ship

CDD should
* help all of us build better conversational AI
* **Save newcomers from habing to learn this the hard way**

### What is the opposite of CDD?

* Developing your assistant for months in isolation before testing it with users
* Only looking at top-level metrics, not conversations themselves
* Autogenerating training data that doesn't reflect ways users really interact
* Pushing changes to production without running tests

### What is CDD?

In 6 steps
1. Share the bot with users
2. Review conversations
3. Annotating user messages and conversations and utterances
4. Testing the bot with automating CI/CD
5. Tracking successes and failures
6. Fixing iteratively to improve.

#### 1. Sharing

* Users will surprise you
* Getting some test users to try prototypes as early as possible!

Shipping without test users never go really well!

#### 2. Reviewing

* At every stage of a project, it is worth reading what users are saying.
* Avoid getting caught up in metrics right away. Conversations are valuable data.

#### 3. Annotating

* Using a script to generate synthetic training data is not good!
* better turning real messages into real training examples 

> Real Conversations > Synthetic Data

During development, training data is created by :
* writing down hypothetical conversations and training examples
* studying existing chat transcripts or knowledge bases
* consulting subject matter experts

These are a great start, but all of those scenarios are based on **human:human** conversations.

> The best training data is generated from the assistant's **actual conversations**

Quality of data, by source (1 to 4, better to worse)

1. Human + bot conversations
2. Human + human conversations
3. Manufactured data
4. Documentation, FAQs

At first, we might want to have a lot of FAQ data and less human/bot conversations because those are not readily available, but over time, we want to have more of human/bot conversations adn less of those manufactured data examples

#### 4. Testing

* Professional teams don't ship applications without tests
* Use whole conversations as end-to-end tests
* Run them on a continuous integration (CI) server

Everything is done through GitHub! THey use GitHub actions to run CI/CD, but we can use anything (e.g. Jenkins, Travis, ...)

#### 5. Tracking

* Use proxy measures to track which conversations are successful and which ones failed
* "Negative" signals are useful too, e.g. users *not* getting back in touch with support

We can create tags from within Rasa X to tag conversations based on different scenarios.

#### 6. Fixing

* Study conversations that went smoothly and ones that failed
* Successful conversations can become new tests !
* Fix issues by annotating more data and/or fixing code


## CDD in Practice

It's not a linear process! you'll find yourself jumping between the different steps!

Some actions require software skills, other need a deep understanding of the user. These are thus not all tackled by the same people!

### CDD for Teams

* Product
    - Product managers
    - innovation and business leaders
* Design
    - UX Designers
    - Conversation Designers
    - Content Creators
* Development
    - Engineers
    - Data Scientists
    - Annotators

steps are done by everyone/some people

## Rasa X

Rasa X is a tool to help teams do CDD

Rasa X turns conversations into training data

Pipeline
```
messaging channels -> conversations -> Rasa X <- Conversations <- Share your bot
```

Rasa X layers on top of Rasa Open Source

* Rasa Open Source : Framework for building AI Assistants
* Rasa X : Tool for Conversation-Driven Development

### Rasa X local mode vs server mode

* Local mode :
    - great for initial testing, familiarizing yourself with Rasa X
    - Conversations get saved to a local db
* Server mode :
    - Great for serving your bot and collecting conversations from many testers or users, 24/7
    - production-ready and scalable
    - Deploys Rasa X (and your assistant) using Docker Compose or Kubernetes
    - Conversations get saved to a production database
    - Includes a Git integration (integrated version control)


We can share our bots with testers using just a link.

Rasa X allows us to review conversations coming from every channel.

We can annotate messages coming in by labeling them as correct, or changing the predicted intent. We can also discard examples using the trash can.

Once we've made some annotations, we can push the new training data to git and trigger the CI pipeline

We can track failures and successes using tags
* Using the API to automatically tag conversations, or add tags manually as we read
* We can turn successful conversations into new end-to-end tests also to make sure the bot always can handle these scenarios.

### Hands-on Rasa X tutorial

Run it locally

## Testing Locally : `ngrok`

Create a secure, publicly accessible tunnel URL connected to a process running on localhost. Allows you to develop as through your application were running on a server instead of locally.

We want to run ngrok once the rasa x server is running, not before because that would give us an error message. running by doing something like :

```bash
./ngrok http 5002

```

## Messaging Channels

* Rasa Open source is connects the NLU and dialogue management to the user through messaging channels
    - slack
    - facebook messenger
    - web chat box
    - google hangouts
    - telegram
    - twilio
    - ...

Messaging channels
* Channel connection details are kept in the `credentials.yml` file
* when you move to production, you'll want to re-create these credentials securely on your server
* after you update credentials.yml, start (or restart) the server with `rasa run`

### Telegram
* Register a new bot
    - go to https://web.telegram.org/#/im?p=@BotFather
    - type `/newbot`
    - Name show up next to the bot's profile icon
    - Username is the bot's handle, and ends in `_bot`

Credentials.yml

```yml
Telegram:
    access_token: "<api_token>"
    verify: "<bot's username>"
    webhook_url: "https://<ngrok-url>/webhooks/telegram/webhook"
```


## What's Next?

Get people to chat with your assistant and keep improving it!

* Where to go from here?
1. Look at conversations and ask : hos does your assistant struggle?
2. Learn how to solve problem from Rasa Docs and Community Forum
3. Improve assistant
4. Test and deploy your updates
5. Repeat

Continually improve your assistant using Rasa X

* *Collect Conversations* between users and your assistant
* *Review conversations* and *improve your assistant* based on what you learn
* Ensure your new assistant passes tests using *Continous Integration (CI)* and redeploy it to users using *Continuous Deployment (CD)*

The path to a contextual assistant

* Quality of assistant is bad without the iterative process of getting user inputs
    - Here we are simply using Rasa Open Source (not Rasa X)
* After a few cycles, we can leverage conversations with Rasa X to manage data more easily
* After a lot of iterations, quality of assistant is going to be really good.

Improve by talking to the assistant yourself, and then giving it to test users, and then real users.









## Resources
* The CCD Playbook blog post : https://blog.rasa.com/the-cdd-playbook-strategies-for-conversation-driven-development/
* Steps to Conversation-Driven Development Youtube Video : https://www.youtube.com/watch?v=7HmDk48KVU8&feature=youtu.be&ab_channel=Rasa
* Conversation-Driven Development Blog Post : https://blog.rasa.com/conversation-driven-development-a-better-approach-to-building-ai-assistants/
