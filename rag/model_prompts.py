# ==================================================================================================================
# Code taken from: HiRAG(https://github.com/2282588541a/HiRAG/tree/main/code) using the LangGPT Framework (https://github.com/langgptai/LangGPT/tree/1e084c181a9b5f251e938ee6c2c24e6407c9f9cc)
# Authors: Xiaoming Zhang, Ming Wang, Xiaocui Yang, Daling Wang, Shi Feng, Yifei Zhang

class Prompt:
    # Prompt which answers a single-hop question using one sentence.
    # Used in query decomposition to extract the answer to a given question from retrieved contexts.
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

    # Prompt which answers a single-hop question using Chain-of-Thought to explain every step of reasoning.
    # Used to answer a multi-hop question given all the retrieved contexts.
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

    # Prompt which detects if retrieved context is sufficient to answer a multi-hop question.
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

    # Prompt which extracts a sub-question from a multi-hop question.
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
Do not ask questions similar to previous questions you have asked.

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

    # Prompt to convert a long answer into a sentence.
    simplify_answer='''
Based on the input and the question, you have to tell me the answer. Answers should be concise and contain only the corresponding keywords.
    Input: The director of the film "Polish-Russian War (Wojna polsko-ruska)" is Xawery Żuławski
    Question: what is the director of film Polish-Russian War (Film)?
    You: Xawery Żuławski

    Input: The mother of the director Xawery Żuławski is Małgorzata Braunek
    Question: Who is the mother of Xawery Żuławski?
    You: Małgorzata Braunek

    Input: Venice's country is Italy while Los Angeles's country is the United States
    Question: Are Venice and Los Angeles in the same country?
    You: No

    Input: Venice's country is Italy while Los Angeles's country is the United States
    Question: Are Venice and Los Angeles in the same country?
    You: No
'''
