# ==================================================================================================================
# Code taken from: HiRAG(https://github.com/2282588541a/HiRAG/tree/main/code) using the LangGPT Framework (https://github.com/langgptai/LangGPT/tree/1e084c181a9b5f251e938ee6c2c24e6407c9f9cc)
# Authors: Xiaoming Zhang, Ming Wang, Xiaocui Yang, Daling Wang, Shi Feng, Yifei Zhang

class Prompt:
    answer_extraction_from_context ='''
    You need to answer a question given some background information.
    Answer the question only if the answer is present in the background information.

    ## Contraints:
    Forget all knowledge you've learned before and only answer the question using the information provided.
    If the background information doesn't contain the answer, say "I don't know".
    Only write one sentence for your answer.
    
    ## Examples:
    User: Who is the director of film Polish-Russian War (Film)?
    User: Polish-Russian War is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.
    You: The director of film Polish-Russian War (Film) is Xawery Żuławski.

    User: When did Apollo 11 land on the moon?
    User: Apollo 11 was a spaceflight conducted by the United States from July 16 to July 24, 1969. It marked the first time in history that humans landed on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC, and Armstrong became the first person to step onto the Moon's surface six hours and 39 minutes later, on July 21 at 02:56 UTC.
    You: Apollo 11 landed on the moon on July 20, 1969, at 20:17 UTC.

    User: What was Shayan Chowdhury Arnob's first musical album?
    User: Arthur Fiedler and the Boston Pops Orchestra recorded the work for RCA Victor, including one of the first stereo recordings of the music.
    You: I don't know.
    '''

    cot_answer_with_context ='''
    You need to answer a question given some background information.
    Answer the question only if the answer is present in the background information.

    ## Contraints:
    Forget all knowledge you've learned before and only answer the question using the information provided.
    If the background information doesn't contain the answer, say "I don't know".
    Explain the reasoning process for your answer in steps using sentences for each step.

    ## Examples:
    User: Who lived longer, Muhammad Ali or Alan Turing?
    User: Muhammad Ali was 74 years old when he died.
    User: Alan Turing was 41 years old when he died.
    You: The context says that Alan Turing lived for 41 years, whilst Muhammad Ali lived for 74 years. 74 is greater than 41, therefore Muhammad Ali lived longer than Alan Turing.
    
    User: Who is the director of the film Polish-Russian War (Film)?
    You: No context was provided for who the director of Polish-Russian War was. I don't know.

    User: How many years elapsed between Abraham Lincoln's birth and the invention of ASCII?
    User: Abraham Lincoln was born on February 12, 1809.
    User: Work on the ASCII standard began in May 1961.
    You: The question is asking for the difference between the year of Abraham Lincoln's birth and the year of invention of ASCII. The context shows that Abrahan Lincoln was born in the year 1809, whilst the ASCII standard began in 1961. 1961 - 1809 = 152, therefore 152 years elapsed between Abraham Lincoln's birth and the creation of the ASCII standard.
    '''

    is_decomposition_needed='''
    ## Answer Deduction Specialist.
    You are an expert at telling whether a question needs follow-up questions to answer or not.
    The user will give you a question and background contexts. Based on the contexts and the question, deduce if the information provided is sufficient to answer the question, or whether follow-up questions are needed to find the answer.
    
    ## Constraints:
    Your answer must be "Yes" if follow-up questions are needed or "No" if the question can be answered from the inputs.
    
    ## Examples:
    Question: Who is the maternal grandfather of Antiochus X Eusebes?
    Context: The mother of Antiochus X Eusebes is Cleopatra IV.
    Context: The father of Cleopatra IV is Ptolemy VIII Physcon.
    Are follow up questions needed here: No.

    Question: Steven Spielberg is from the United States.
    Context: Are both the directors of Jaws and Casino Royale from the same country?
    Context: The director of Jaws is Steven Spielberg.
    Are follow up questions needed here: Yes.

    Question: Martin Campbell is from New Zealand.
    Context: Are both the directors of Jaws and Casino Royale from the same country?
    Context: The director of Jaws is Steven Spielberg.
    Context: Steven Spielberg is from the United States.
    Context: The director of Casino Royale is Martin Campbell.
    Are follow up questions needed here: No.

    Question: Who lived longer, Muhammad Ali or Alan Turing?
    Context: Muhammad Ali was 74 years old when he died.
    Are follow up questions needed here: Yes.

    Question: Who lived longer, Muhammad Ali or Alan Turing?
    Context: Muhammad Ali was 74 years old when he died.
    Context: Alan Turing was 41 years old when he died.
    Are follow up questions needed here: No.
    '''

    query_decomposer='''
    ## Question Decomposition Specialist:
    You are an expert at breaking down difficult problems into simple problems by analysing them.
    The user needs you to answer a complex question for them which is hard to answer directly.
    Answer the question by decomposing it and tell the user at the right time when the problem can be solved.

    ## Constraints:
    Forget all knowledge you've learned before and decompose the user's question based only on the user's answers.
    To make it easier for the user to answer, only ask one simple question at a time.
    You can only decompose the question, do not answer it directly.
    Only write one sentence for your answer containing a simple question.

    ## Examples:
    User: What is the award that the director of film Wearing Velvet Slippers Under A Golden Umbrella won?
    You: Who is the director of Wearing Velvet Slippers Under A Golden Umbrella?
    User: The director of Wearing Velvet Slippers Under A Golden Umbrella is Maung Wunna.
    You: What awards has Maung Wunna won?
    User: Maung Wunna won the Myanmar Motion Picture Academy Awards.
    You: That's enough.
    
    User: Which film's director died earlier, Condemned Women or Faces In The Dark?
    You: What is the director of the film Condemned Women?
    User: The director of the film Condemned Women is Lew Landers.
    You: What is the director of the film Faces In The Dark?
    User: The director of the film Faces In The Dark is David Eady.
    You: When did Lew Landers die?
    User: Lew Landers died on 16 December 1962.
    You: When did David Eady die?
    User: David Eady died on April 5, 2009.
    You: That's enough.
    '''
    ans_prompt='''you should answer the question with the konwn information .You should first analyze the question and the konwn information given and finally give the answer.Let's think step by step
    Question: Who is the mother of the director of film Polish-Russian War (Film)?
    ##konwn information:The director of Polish-Russian War is Xawery Żuławski., Xawery Žuławski's mother is Małgorzata Braunek.
    You: Step 1: Analyze the Question 
    The question asks for the mother of the director of the film "Polish-Russian War (Film)." 

    Step 2: Analyze the Known Information
    We know that the director of the film "Polish-Russian War (Film)" is Xawery Żuławski. Additionally, we are given that Xawery Żuławski's mother is Małgorzata Braunek.

    Step 3: Answer the Question
    Based on the known information, the mother of the director of the film "Polish-Russian War (Film)" is Małgorzata Braunek.
    Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
    ##konwn information:the publication date of Blind Shaft is 2003., the publication date of The Mask Of Fu Manchu is 1932.
    You: Step 1: Analyze the Question
    The question asks which film, "Blind Shaft" or "The Mask Of Fu Manchu," was released first.

    Step 2: Analyze the Known Information
    We know that "Blind Shaft" was released in 2003, and "The Mask Of Fu Manchu" was released in 1932.

    Step 3: Answer the Question
    Based on the known information, "The Mask Of Fu Manchu" came out first, in 1932, while "Blind Shaft" was released in 2003.

    Question: '''
    is_ans_prompt='''I will tell you the question,correct answer and response. You need to judge whether the response is correct.If the answer is correct ,return yes,else,return no. Examples are as follows:
    question:Who is the mother of the director of film Polish-Russian War (Film)?
    correct answer:Jagna Žuławski
    response:Jagna Žuławski
    output:yes
    '''
    exact_prompt='''Based on the input , you have to find the answer,which usually on the behind of the "Answer:"
    input:1. Analyzing the Question:
    - The question seeks to identify the director of the film "Polish-Russian War (Wojna polsko-ruska)."
    - The known information provided is that the film was directed by Xawery Żuławski.
    - Additionally, it's mentioned that the film is based on the novel "Polish-Russian War under the white-red flag" by Dorota Masłowska.

    2. Known Information:
    - The director of the film is Xawery Żuławski.
    - The film is based on the novel by Dorota Masłowska.

    3. Answer:
    - The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski.
    output: The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski.
    input: 1: Analyze the Question
    The question asks which film, "Blind Shaft" or "The Mask Of Fu Manchu," was released first.

    Step 2: Analyze the Known Information
    We know that "Blind Shaft" was released in 2003, and "The Mask Of Fu Manchu" was released in 1932.

    Step 3: Answer the Question
    Based on the known information, "The Mask Of Fu Manchu" came out first, in 1932, while "Blind Shaft" was released in 2003.
    output: Based on the known information, "The Mask Of Fu Manchu" came out first, in 1932, while "Blind Shaft" was released in 2003.

    '''
    exact_prompt2='''Based on the input and the question, you have to tell me the answer.Answers should be concise and contain only the corresponding keywords
    Input: The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski
    Question: what is the director of film Polish-Russian War (Film)?
    You: Xawery Żuławski

    Input: The director of Xawery Żuławski is Małgorzata Braunek
    Question: Who is the mother of Xawery Żuławski?
    You: Małgorzata Braunek

    Input: '''
    exact_prompt3='''Based on the input and the question, you have to tell me the answer.Answers should be concise and contain only the corresponding keywords
    Input: The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski
    Question: what is the director of film Polish-Russian War (Film)?
    You: Xawery Żuławski

    Input: The director of Xawery Żuławski is Małgorzata Braunek
    Question: Who is the mother of Xawery Żuławski?
    You: Małgorzata Braunek

    Input: Venice's country is Italy while Los Angeles's country is the United States
    Question: Are Venice and Los Angeles in the same country?
    You: No

    Input: Venice's country is Italy while Los Angeles's country is the United States
    Question: Are Venice and Los Angeles in the same country?
    You: No

    Input: '''
    ret_prompt='''you should answer the question with the konwn information .You should first analyze the question and the konwn information given and finally give the answer.Let's think step by step!
    Question: Who is the director of film Polish-Russian War (Film)?
    ##konwn information:Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska.
    You: 1. Analyzing the Question:
    - The question seeks to identify the director of the film "Polish-Russian War (Wojna polsko-ruska)."
    - The known information provided is that the film was directed by Xawery Żuławski.
    - Additionally, it's mentioned that the film is based on the novel "Polish-Russian War under the white-red flag" by Dorota Masłowska.

    2. Known Information:
    - The director of the film is Xawery Żuławski.
    - The film is based on the novel by Dorota Masłowska.

    3. Answer the Question:
    - The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski.

    Question: Who is the mother of Xawery Żuławski?
    ##konwn information:Xawery Żuławski (born 22 December 1971 in Warsaw) is a Polish film director.

    In 1995 he graduated National Film School in Łódź. He is the son of actress Małgorzata Braunek and director Andrzej Żuławski. His second feature Wojna polsko-ruska (2009), adapted from the controversial best-selling novel by Dorota Masłowska, won First Prize in the New Polish Films competition at the 9th Era New Horizons Film Festival in Wrocław. In 2013, he stated he intends to direct a Polish novel "Zły" by Leopold Tyrmand.
    You: 1. Analyzing the Question:
    - The question seeks to identify the mother of Xawery Żuławski.
    - The known information provided includes Xawery Żuławski's birthdate, occupation as a Polish film director, and details about his education and career.
    - It's mentioned that his mother is an actress named Małgorzata Braunek and his father is a director named Andrzej Żuławski.

    2. Known Information:
    - Xawery Żuławski was born on December 22, 1971, in Warsaw, Poland.
    - He is a Polish film director.
    - His mother is actress Małgorzata Braunek.
    - His father is director Andrzej Żuławski.

    3. Answer the Question:
    - The mother of Xawery Żuławski is Małgorzata Braunek.


    '''

    revise_prompt=''' you are given a question ,some information and a subquestion. the subquestion may have some fault,you need to correct it .Examples are as follows:
    Question: Who is the mother of the director of film Polish-Russian War (Film)?
    ##konwn information:The director of Polish-Russian War is Xawery Żuławski.
    ##subquestion:Who is the mother of Xawery Żułwski?
    You: Who is the mother of Xawery Żuławski?


    '''
    choose_prompt='''Based on the ithe question and information, you must return yes or no.
    remeber:you must return yes or no
    '''
    google_entity_prompt='''You need to describe an entity in a information.I will give you the entity and information,You need to describe an entity based the information .examples are as follows:
    ##entity:Xawery Żuławski
    ##information:The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski.
    You: Xawery Żuławski is the director of the film "Polish-Russian War (Wojna polsko-ruska)".

    ##entity:Małgorzata Braunek.
    ##information:The mother of Xawery Żuławski is Małgorzata Braunek.
    You: Małgorzata Braunek is the mother of Xawery Żuławski.



    '''

    exact_prompt4='''You need to extract the answer to the question from the reply. Note that only the part related to the answer is retained.
    Question: Who is the director of film Polish-Russian War (Film)?
    ##reply:The director of the film "Polish-Russian War" is Dziga Vertov. Released in 1920, it's a Soviet silent documentary film detailing the Polish-Soviet War.Sorry,I am an artificial intelligence and do not have real-time information
    You: The director of the film "Polish-Russian War" is Dziga Vertov

    Question: '''
    ques_prompt='''I will give you a question and you need to return the answer,examples are as follows:
    Question: what is the date of birth of Don Chaffey?
    You: the date of birth of Don Chaffey is August 5, 1917.

    Question: what is the director of The Half-Way Girl?
    You: the director of The Half-Way Girl is John Francis Dillon.


    ##question'''
    can_answer_prompt1='''Based on the known infotmation and question,You need to tell me if you can answer the question or not.If you can answer the question,return yes with answer,else return no. 
    Question: Who is the mother of the director of film Polish-Russian War (Film)?
    ##konwn information:The director of Polish-Russian War is Xawery Żuławski., Xawery Žuławski's mother is Małgorzata Braunek.
    You: yes, Małgorzata Braunek

    Question: Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
    ##konwn information:the publication date of Blind Shaft is 2003., the publication date of The Mask Of Fu Manchu is 1932.
    You: yes, The Mask Of Fu Manchu

    Question: When did John V, Prince Of Anhalt-Zerbst's father die?
    ##konwn information:the fatherJohn V of Anhalt-Zerbst is Ernest I, Prince of Anhalt-Dessau
    ##ouput:no

    #question:Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?
    ##konwn information:the father of Charles Bretagne Marie de La Trémoille is Jean Bretagne Charles de La Trémoille.the father of Jean Bretagne Charles de La Trémoille is Charles Armand René de La Trémoille.
    You: yes, Charles Armand René de La Trémoille

    Question: '''
    can_answer_prompt2='''Based on the question and a response from others, you have to tell me if the response can answer the question. Your answer must be yes or no.
    Question:  What is the date of death of Armin, Prince Of Lippe's father?
    ##response: Based on the known information, the date of death of Armin, Prince Of Lippe's father, Leopold IV, Prince of Lippe, is December 30, 1949.
    You:  yes

    Question:  Which film has the director died earlier, Love In Exile or Manchi Vallaki Manchivadu?
    ##response: Answer: Unable to determine.
    You:  no

    Question:  Who is the paternal grandfather of Zubdat-Un-Nissa?
    ##response: Based on the known information, the paternal grandfather of Zubdat-Un-Nissa is Shah Jahan.
    You:  yes

    Question:  '''