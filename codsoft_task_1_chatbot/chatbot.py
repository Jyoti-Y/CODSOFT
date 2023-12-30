import time
import random
import tkinter as tk

greetings = ['hello', 'hi', 'hey!','hey']
keywords = ['weather', 'time', 'date', 'day', 'temperature', 'rain', 'sun', 'cloud', 'clouds', 'cloudy', 'wind', 'windy', 'snow', 'snowy', 'rainy', 'sunny', 'hot', 'cold', 'warm', 'freezing', 'boiling', 'freezing', 'warm', 'boiling']
intensions = {
    'question': ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whom', 'whose', 'tell'],
    'positive': ['yes', 'yep', 'yeah', 'sure', 'ok', 'okay', 'fine', 'cool', 'understood', 'understand','great', 'awesome', 'fantastic', 'good', 'nice', 'sweet', 'yay'],
    'negative': ['no', 'nope', 'nah', 'not', 'hate', 'sad', 'bad', 'awful', 'terrible', 'never', 'dont', 'do not', 'dont know', 'do not know', 'dont understand', 'do not understand', 'dont get it', 'do not get it', 'dont know what you mean', 'do not know what you mean', 'dont know what you are saying', 'do not know what you are saying', 'dont know what you are talking about', 'do not know what you are talking about', 'dont know what you are talking', 'do not know what you are talking', 'dont know what you are talking about', 'do not know what you are talking about', 'dont know what you are talking', 'do not know what you are talking', 'dont know what you mean', 'do not know what you mean', 'dont know what you are saying', 'do not know what you are saying', 'dont know what you are talking about', 'do not know what you are talking about', 'dont know what you are talking', 'do not know what you are talking', 'dont know what you are talking about', 'do not know what you are talking about', 'dont know what you are talking', 'do not know what you are talking'],
    'thanks': ['thanks', 'thank you', 'thx', 'thnx', 'ty', 'thank you very much', 'thank you so much', 'thanks a lot', 'thanks a bunch', 'thanks a ton', 'thanks heaps', 'thanks loads', 'thanks loads'],
    'queryForWhatChatBotCanDo': ['can you help me', 'how can you help me', 'what are your functions', 'what are your capabilities', 'please help me'],
    'Safety': ['can i go out', 'should i go out', 'is it safe out', 'is it safe to go out', 'is it safe to go outside', 'is it safe to go out', 'is it safe to go out today', 'is it safe to go out tomorrow', 'is it safe to go out tonight', 'is it safe to go out this evening', 'is it safe to go out this morning', 'is it safe to go out this afternoon', 'is it safe to go out now', 'is it safe to go out today', 'is it safe to go out tomorrow', 'is it safe to go out tonight', 'is it safe to go out this evening', 'is it safe to go out this morning', 'is it safe to go out this afternoon', 'is it safe to go out now', 'is it safe to go out today', 'is it safe to go out tomorrow', 'is it safe to go out tonight', 'is it safe to go out this evening', 'is it safe to go out this morning', 'is it safe to go out this afternoon', 'is it safe to go out now', 'is it safe to go out today', 'is it safe to go out tomorrow', 'is it safe to go out tonight', 'is it safe to go out this evening', 'is it safe to go out this morning', 'is it safe to go out this afternoon', 'is it safe to go out now', 'is it safe to go out today', 'is it safe to go out tomorrow', 'is it safe to go out tonight', 'is it safe to go out this evening', 'is it safe to go out this morning', 'is it safe to go out this afternoon', 'is it safe to go out now', 'is it safe to go out today', 'is it safe to go out tomorrow', 'is it safe to go out tonight', 'is it safe to go out this evening', 'is it safe to go out this morning', 'is it safe to go out this afternoon', 'is it safe to go out now'],
    'firstPerson':['i', 'me', 'my', 'mine', 'myself'],
    'secondPerson':['you', 'your', 'yours', 'yourself'],
    'thirdPerson':['he', 'she', 'it', 'him', 'her', 'his', 'hers', 'its', 'they', 'them', 'their', 'theirs', 'himself', 'herself', 'itself', 'themselves'],
}
bye = ['bye', 'goodbye', 'stop', 'see you later', 'see you soon', 'talk to you later', 'talk to you soon']
weathers = ['sunny', 'cloudy', 'rainy', 'windy', 'snowy', 'hot', 'cold', 'warm', 'freezing', 'boiling']

responses = {
    'greetings': ['Hello! How may I help you', 'Hi!', 'Hey! How are you?'],
    'question': ['I am a chatbot. I can help you with weather and time related queries.'],
    'queryForWhatChatBotCanDo': ['I am a chatbot. I can help you with weather and time related queries.'],
    'positive': ['Great!', 'Awesome!', 'Fantastic!', 'Good!', 'Nice!', 'Sweet!', 'Yay!'],
    'negative': ['Oh no!', 'Oh dear!', 'Oh my!', 'Oh gosh!', 'Oh wow!', 'Oh jeez!', 'sad to hear that'],
    'thanks': ['You are welcome!', 'No problem!', 'No worries!', 'My pleasure!', 'Anytime!'],
    'safety': ['It is safe to go out.'],
    'time': ['The time is ' + time.strftime("%H:%M:%S")],
    'date': ['The date is ' + time.strftime("%d/%m/%Y")],
    'day': ['Today is ' + time.strftime("%A")],
    'weather': ['The weather today is ' + random.choice(weathers)],
    'randomResponse': ['I am not sure what you mean. Can you please rephrase that?', 'I am not sure what you are saying. Can you please rephrase that?', 'I am not sure what you are talking about. Can you please rephrase that?', 'I am not sure what you mean. Can you please rephrase that?', 'I am not sure what you are saying. Can you please rephrase that?', 'I am not sure what you are talking about. Can you please rephrase that?', 'I am not sure what you mean. Can you please rephrase that?', 'I am not sure what you are saying. Can you please rephrase that?', 'I am not sure what you are talking about. Can you please rephrase that?'],
    'askForhealth':['How are you feeling today?'],
    'answerAboutHealth':['I am great'],
    'bye': ['Bye!', 'Goodbye!', 'See you later!', "I'll miss you!", 'Have a nice day!', 'Pleasure talking to you!']
}

def send_message():
    user_input = user_entry.get() 
    user_entry.delete(0, tk.END)  
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + user_input + "\n")
    chat_log.config(state=tk.DISABLED)  
    respond_to_user(user_input)


def respond_to_user(user_input):
    flags = []
    exit = False
    for sentence in user_input.split(','):
        flag = []
        for word in sentence.split(' '):
            if word in greetings:
                flag.append('greetings')
            elif word in keywords:
                flag.append(word)
            for intension in intensions:
                if word in intensions[intension]:
                    flag.append(intension)
            if word in bye:
                flag.append('bye')
        flags.append(flag)
        for intension in intensions:
            if sentence in intensions[intension]:
                flag.append(intension)
                break
    responded = False
    chatbot_response = "chatbot: "
    for respond in flags:
        if 'greetings' in respond:
            chatbot_response += random.choice(responses['greetings']) + "\n"
            responded = True
        elif 'question' in respond:
            if 'whatAreYou' in respond:
                chatbot_response += random.choice(responses['whatAreYou']) + "\n"
            elif 'weather' in respond:
                chatbot_response += random.choice(responses['weather']) + "\n"
            elif 'time' in respond:
                chatbot_response += random.choice(responses['time']) + "\n"
            elif 'date' in respond:
                chatbot_response += random.choice(responses['date']) + "\n"
            elif 'day' in respond:
                chatbot_response += random.choice(responses['day']) + "\n"
            elif 'secondPerson' in respond:
                chatbot_response += random.choice(responses['answerAboutHealth']) + "\n"
                chatbot_response += random.choice(responses['askForhealth']) + "\n"
            responded = True
        elif 'Safety' in respond:
            chatbot_response += random.choice(responses['safety']) + "\n"
            responded = True
        elif 'positive' in respond and not 'question' in respond:
            chatbot_response += random.choice(responses['positive']) + "\n"
            responded = True
        elif 'negative' in respond and not 'question' in respond:
            chatbot_response += random.choice(responses['negative']) + "\n"
            responded = True
        elif 'thanks' in respond:
            chatbot_response += random.choice(responses['thanks']) + "\n"
            responded = True
        elif 'safety' in respond:
            chatbot_response += random.choice(responses['safety']) + "\n"
            responded = True
        elif 'time' in respond:
            chatbot_response += random.choice(responses['time']) + "\n"
            responded = True
        elif 'date' in respond:
            chatbot_response += random.choice(responses['date']) + "\n"
            responded = True
        elif 'day' in respond:
            chatbot_response += random.choice(responses['day']) + "\n"
            responded = True
        elif 'bye' in respond:
            chatbot_response += random.choice(responses['bye']) + "\n"
            responded = True
            exit = True
        elif 'askForhealth' in respond:
            chatbot_response += random.choice(responses['askForhealth']) + "\n"
            responded = True
        elif 'queryForWhatChatBotCanDo' in respond:
            chatbot_response = "Chatbot: " + random.choice(responses['queryForWhatChatBotCanDo']) + "\n"
            break

    if not responded:
        chatbot_response += random.choice(responses['randomResponse']) + "\n"
    if exit:
        root.destroy()
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, chatbot_response)
    chat_log.config(state=tk.DISABLED)


root = tk.Tk()
root.title("Chatbot")

chat_log = tk.Text(root, state=tk.DISABLED, wrap=tk.WORD)
chat_log.pack()

user_entry = tk.Entry(root)
user_entry.pack()

root.bind("<Return>", lambda x: send_message())

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

root.mainloop()