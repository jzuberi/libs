from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

import concurrent.futures
import functools
import logging

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
import requests, time, re

from langchain_core.runnables import RunnableLambda

"""
deepseek-r1
"""

local_llm = 'llama3.1:latest'
lmstudio_base_url = "http://localhost:1234/v1"


class TimeoutRunnable:
    def __init__(self, runnable, timeout=10):
        self.runnable = runnable
        self.timeout = timeout

    def invoke(self, inputs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.runnable.invoke, inputs)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                return {"answer": "timeout"}
            except Exception as e:
                print(e)
                return {"answer": "error"}
            

def lmstudio_is_serving(model_name):
    
    is_serving = False
    
    url = lmstudio_base_url + "/chat/completions"
    headers = {"Authorization": "Bearer lmstudio"}
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Are you ready?"}],
        "temperature": 0.5
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        is_serving=True

    except requests.exceptions.RequestException as e:
        
        exception = e
        
    return(is_serving)


def parse_llm_resp(funcResp, keyName, valType=str, delay_secs=2):

    parsedResp = None

    is_dict = type(funcResp) is dict
    
    if(is_dict):
        
        has_key = keyName in funcResp.keys()
        
        if(has_key):
            has_str_val = type(funcResp[keyName]) is valType
            
            if(has_str_val):

                parsedResp = funcResp[keyName]

    time.sleep(delay_secs)

    return(parsedResp)

def parse_llm_resp_multi(funcResp, key_types: dict, delay_secs=2):
    """
    funcResp: dict returned by LLM pipeline
    key_types: dict mapping key -> expected Python type
               e.g. {"title": str, "summary_short": str, "summary_long": str}

    Returns: dict of parsed values or None for missing/invalid keys
    """

    parsed = {}

    if isinstance(funcResp, dict):
        for key, expected_type in key_types.items():
            if key in funcResp and isinstance(funcResp[key], expected_type):
                parsed[key] = funcResp[key]
            else:
                parsed[key] = None  # missing or wrong type
    else:
        # funcResp was not a dict at all
        for key in key_types.keys():
            parsed[key] = None

    time.sleep(delay_secs)
    return parsed


def get_entity_category(ent_cat_dict, local_llm='llama3.2'):

    is_lmstudio_model = lmstudio_is_serving(local_llm)

    if(is_lmstudio_model is False):

        llm_json_out = ChatOllama(
            model=local_llm, 
            format="json", 
            temperature=0, 
            stop=["<|eot_id|>"], 
            timeout=10, 
            num_predict= 3000
        )

        template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are a senior taxonomy/ontology analyst. 
            Your job is to categorize a """ + ent_cat_dict['entity_type'] + """ 
            grouping. You are provided with a specific context about 
            this group of """ + ent_cat_dict['entity_type'] + """s
            The goal is to provide a succinct categorization with 
            """ + ent_cat_dict['cat_length'] + """.
            For example if """ + ent_cat_dict['cat_example'] + """.
            If there is not enough context information to answer the question,
            respond with 'not enough information'. Provide the english language answer 
            as a JSON with a single key 'category' 
            and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the """ + ent_cat_dict['entity_type'] + """ group: \n\n {val} \n\n
            Here is the context: \n\n {val_context} \n\n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    else:

        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # LM Studio's default API endpoint
            api_key="lmstudio",  # Dummy key; LM Studio doesn't require auth
            model=local_llm,
            temperature=0,
            timeout=10
        )

        template_str = """
            You are a senior taxonomy/ontology analyst. 
            Your job is to categorize a """ + ent_cat_dict['entity_type'] + """ 
            grouping. You are provided with a specific context about 
            this group of """ + ent_cat_dict['entity_type'] + """s. 
            The goal is to provide a succinct categorization with 
            """ + ent_cat_dict['cat_length'] + """.
            For example if """ + ent_cat_dict['cat_example'] + """.
            If there is not enough context information to answer the question,
            respond with 'not enough information'. Provide the english language answer 
            as a JSON with a single key 'category' 
            and no preamble or explanation.
            Here is the """ + ent_cat_dict['entity_type'] + """ group: \n\n {val} \n\n
            Here is the context: \n\n {val_context} \n\n
            """

    entity_cat_prompt = PromptTemplate(
        template=template_str,
        input_variables=['val','val_context'],
    )

    entity_cat = entity_cat_prompt | llm_json_out | JsonOutputParser()
    entity_cat = entity_cat.with_retry()
    
    return(entity_cat)


def strip_think_blocks(text):
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also remove any leading/trailing whitespace
    return text.strip()

def extract_json(text):
    # find the first JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text  # let JsonOutputParser throw the error
    return text[start:end+1]



def get_general_answerer(local_llm = 'llama-3.1-8b-instruct', retry=True, local_timeout=10):

    is_lmstudio_model = lmstudio_is_serving(local_llm)

    if(is_lmstudio_model is False):

        llm_json_out = ChatOllama(
            model=local_llm, 
            format="json", 
            temperature=0, 
            stop=["<|eot_id|>"], 
            timeout=local_timeout, 
            num_predict= 3000
            )

        template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are a thoughtful and strategic expert. 
            You are provided with the following information: 
            (1) Question 
            For instance, if provided with a question about how to choose a hobby,
            give helpful and thoughtful suggestions.
            Provide the english language answer 
            in a JSON with a single key 'answer' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the text of the question: \n\n {question} \n\n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    else:

        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # LM Studio's default API endpoint
            api_key="lmstudio",  # Dummy key; LM Studio doesn't require auth
            model=local_llm,
            temperature=0.1,
            timeout=local_timeout
        )

        template_str = """
            You are a thoughtful and strategic expert. 
            You are provided with the following information: 
            (1) Question 
            For instance, if provided with a question about how to choose a hobby,
            give helpful and thoughtful suggestions.
            Provide the english language answer 
            in a JSON with a single key 'answer' and no preamble, no explanation.
            Here is the text of the question: \n\n {question} \n\n
            """

    llm_gen_q_prompt = PromptTemplate(
        template=template_str,
        input_variables=['question'],
    )

    cleaner = StrOutputParser()  # ensures we get raw text first

    gen_q_func = (
        llm_gen_q_prompt
        | llm_json_out
        | cleaner
        | (lambda text: strip_think_blocks(text))
        | extract_json
        | JsonOutputParser()
    )

    if(retry):
        
        gen_q_func = gen_q_func.with_retry()

    
    return TimeoutRunnable(gen_q_func, timeout=local_timeout)


def get_metadata_answerer(local_llm='llama-3.1-8b-instruct', retry=True, local_timeout=10):

    is_lmstudio_model = lmstudio_is_serving(local_llm)

    if not is_lmstudio_model:
        llm_json_out = ChatOllama(
            model=local_llm,
            format="json",
            temperature=0,
            stop=["<|eot_id|>"],
            timeout=local_timeout,
            num_predict=3000
        )

        template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a careful and precise metadata generator.
        You must return valid JSON with the keys:
        "title", "summary_short", "summary_long", "source_url".
        No preamble, no explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the text of the question:

        {question}

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    else:
        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lmstudio",
            model=local_llm,
            temperature=0.1,
            timeout=local_timeout
        )

        template_str = """
        You are a careful and precise metadata generator.
        You must return valid JSON with the keys:
        "title", "summary_short", "summary_long", "source_url".
        No preamble, no explanation.

        Here is the text of the question:

        {question}
        """

    llm_prompt = PromptTemplate(
        template=template_str,
        input_variables=['question'],
    )

    cleaner = StrOutputParser()

    gen_meta_func = (
        llm_prompt
        | llm_json_out
        | cleaner
        | (lambda text: strip_think_blocks(text))
        | extract_json
        | JsonOutputParser()
    )

    if retry:
        gen_meta_func = gen_meta_func.with_retry()

    return TimeoutRunnable(gen_meta_func, timeout=local_timeout)


def get_llm(local_llm='llama-3.1-8b-instruct', retry=True, local_timeout=10):

    is_lmstudio_model = lmstudio_is_serving(local_llm)

    if not is_lmstudio_model:
        llm_json_out = ChatOllama(
            model=local_llm,
            format="json",
            temperature=0,
            stop=["<|eot_id|>"],
            timeout=local_timeout,
            num_predict=3000
        )

        template_str = """
        {question}
        """

    else:
        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lmstudio",
            model=local_llm,
            temperature=0.1,
            timeout=local_timeout
        )

        template_str = """
        {question}
        """

    llm_prompt = PromptTemplate(
        template=template_str,
        input_variables=['question'],
    )

    cleaner = StrOutputParser()

    gen_meta_func = (
        llm_prompt
        | llm_json_out
        | cleaner
        | (lambda text: strip_think_blocks(text))
        | extract_json
        | JsonOutputParser()
    )

    if retry:
        gen_meta_func = gen_meta_func.with_retry()

    return TimeoutRunnable(gen_meta_func, timeout=local_timeout)




def get_content_transformer(
        translate_dict, 
        local_model = 'llama3.2', 
        inputName='article content'
        ):
    
    llm_json_out = ChatOllama(
        model=local_model, 
        format="json", 
        temperature=0, 
        stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000
        )
    
    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        """ + translate_dict['character_desc'] + """.
        Keep text """ + translate_dict['tone_desc'] + """.
        You produce """ + translate_dict['output_desc_informal'] + """. 
        You are provided with the following information: 
        (1) """ + inputName + """ 
        Provide the """ + translate_dict['output_desc_formal'] + """
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the """ + inputName + """: \n\n {article_content} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_ctrans_prompt = PromptTemplate(
        template=template_str,
        input_variables=[inputName],
    )

    content_trans_func = llm_ctrans_prompt | llm_json_out | JsonOutputParser()
    content_trans_func = content_trans_func.with_retry()
    
    return(content_trans_func)

def get_text_classifier(geo_dict = {}, local_model = 'llama3.1:latest', is_strict=True):

    is_lmstudio_model = lmstudio_is_serving(local_model)

    if(is_lmstudio_model is False):

        llm_json_out = ChatOllama(
        model=local_model, 
        format="json", 
        temperature=0, 
        stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000
        )

        template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are an analyst who determines if an article relates to """ + geo_dict['text_desc'] + """. 
            You are provided with the following information: 
            (1) article content 
            For instance, """ + geo_dict['text_example'] + """. 
            """ + (is_strict==True)*"""This evaluation should be a strict test.""" + """
            Provide the english language answer 
            in a JSON with a single key '""" + geo_dict['key_name'] + """' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the text of the article: \n\n {article_content} \n\n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    else:

        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # LM Studio's default API endpoint
            api_key="lmstudio",  # Dummy key; LM Studio doesn't require auth
            model=local_model,
            temperature=0,
            timeout=10
        )

        template_str = """
        You are an analyst who determines if an article relates to """ + geo_dict['text_desc'] + """. 
            You are provided with the following information: 
            (1) article content 
            For instance, """ + geo_dict['text_example'] + """. 
            """ + (is_strict==True)*"""This evaluation should be a strict test.""" + """
            Provide the english language answer 
            in a JSON with a single key '""" + geo_dict['key_name'] + """' and no preamble or explanation.
            Here is the text of the article: \n\n {article_content} \n\n
            """

    

    llm_is_us_prompt = PromptTemplate(
        template=template_str,
        input_variables=['article_content'],
    )

    is_us_func = llm_is_us_prompt | llm_json_out | JsonOutputParser()
    is_us_func = is_us_func.with_retry()

    return is_us_func

def get_has_characteristics(characteristics_dict = {}, local_model = 'llama3.1:latest', is_strict=True):

    llm_json_out = ChatOllama(
        model=local_model, 
        format="json", 
        temperature=0, 
        stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000
        )

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an analyst who determines if an article has certain characteristics.
        Specifically, """ + characteristics_dict['desc'] + """. 
        You are provided with the following information: 
        (1) article content 
        """ + characteristics_dict['additional_info'] + """. 
        """ + (is_strict==True)*"""This evaluation should be a strict test.""" + """
        Provide the answer 
        in a JSON with a single key '""" + characteristics_dict['key_name'] + """' and no preamble or explanation.
        The values should be True or False.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the text of the article: \n\n {article_content} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_has_characteristics_prompt = PromptTemplate(
        template=template_str,
        input_variables=['article_content'],
    )

    has_characteristics_func = llm_has_characteristics_prompt | llm_json_out | JsonOutputParser()
    has_characteristics_func = has_characteristics_func.with_retry()

    return(has_characteristics_func)





def get_text_parser(geo_dict = {}, local_model = 'llama3.2'):

    llm_json_out = ChatOllama(
        model=local_model, 
        format="json", 
        temperature=0, 
        stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000
    )

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an analyst who extracts """ + geo_dict['val_type'] + """ information such as """ + geo_dict['val_desc'] + """
        from text. 
        You are provided with the following information: 
        (1) article content 
        For instance, if the text """ + geo_dict['val_example'] + """. 
        This evaluation should be strict. """ + geo_dict['val_exceptions'] + """
        Provide the english language answer 
        in a JSON with a single key '""" + geo_dict['key_name'] + """' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the text of the article: \n\n {article_content} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_extract_geo_prompt = PromptTemplate(
        template=template_str,
        input_variables=['article_content'],
    )

    extract_geo_func = llm_extract_geo_prompt | llm_json_out | JsonOutputParser()
    extract_geo_func = extract_geo_func.with_retry()

    return(extract_geo_func)

def get_title_is_specific(llm_model = 'llama3.2'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_title_is_specific = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing the specificity of a proposed title to an article. 
        If the proposed title has specific information 
        (e.g. company names, people names, locations, amounts, acronyms, etc.), 
        grade it as specific. Otherwise grade it as not specific. 
        It does not need to be a stringent test. 
        The goal is to filter out overly general article titles. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document 
        is specific. \n
        Provide the binary score as a JSON with a single key 'is_specific' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the proposed title: \n\n {proposed_title} \n\n
        Here is the article content: {content} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["proposed_title", "content"],
    )

    title_is_specific = prompt_title_is_specific | llm_json_out | JsonOutputParser()
    title_is_specific = title_is_specific.with_retry()
    
    return(title_is_specific)


def get_title_specified(llm_model = 'llama3.1'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_title_specifier = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an editor who revises article titles to make them engaging by adding additional
        details and names. 
        Your goal is to update the proposed article title to incorporate additional details from the 
        article content (e.g. company names, people names, product names, locations, amounts, acronyms, etc.). 
        If there is no specificity in the article content, the new title should be an empty string.
        The goal is to make the article title more detailed and engaging than the proposed title; 
        while still reflecting the content of the article.
        Provide the updated article title as a JSON with a key 'title' if there is no
        additional specificity and details in the content, write an empty string.
        The updated title should be less than 10 words and more detailed than the proposed title.
        No preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the proposed title: \n\n {proposed_title} \n\n
        Here is the article content: {content} \n 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["proposed_title", "content"],
    )

    title_specifier = prompt_title_specifier | llm_json_out | JsonOutputParser()
    title_specifier = title_specifier.with_retry()
    
    return(title_specifier)

def get_source_grader(llm_model = 'llama3.1'):

    llm_json_out = ChatOllama(
        model=llm_model, 
        format="json", 
        temperature=0, stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000
        )

    prompt_source_grader = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing the relevance of a potential source to an article. 
        If the proposed source is related to the article content, grade it as relevant.
        Otherwise grade it as not relevant. 
        The goal is to filter out irrelevant sources that do not relate to the article content. \n
        Give a binary score 'yes' or 'no' score to indicate whether the source 
        is relevant. \n
        Provide the binary score as a JSON with a single key 'is_source' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the proposed source: \n\n {proposed_source} \n\n
        Here is the article content: {content} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["proposed_source", "content"],
    )

    source_grader = prompt_source_grader | llm_json_out | JsonOutputParser()
    source_grader = source_grader.with_retry()
    
    return(source_grader)

def get_article_to_prompter(llm_model = 'llama3.1', remove_entities=False):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)
    
    pa_prompt0 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a
        creative category editor who comes up with coherent ideas for new articles based on rough notes 
        from your correspondents in the field. By reviewing the notes,
        you write concise, detailed descriptions for new articles. For example, if you are a category
        editor for the arts section and you read correspondent notes about street art in Asia,
        you come up with a description like "street art in asia like graffiti and murals."
        Article descriptions should be specific enough to guide writers, not something broad like "art in asia."
        Article descriptions should have these characteristics:
        1. be 20-30 words in length.
        2. relate to the category name.
        3. relate to the note provided.
        Provide the 20 to 30 word article description as a JSON with a key 'article_desc' for the description
        and a key 'kws' for a list of keywords to help writers find more information (also based on the note).
        No preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the correspondent note: \n\n {note} \n\n
        Here is the category name: {category} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    pa_prompt1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a
        creative category editor who comes up with coherent ideas for new articles based on rough notes 
        from your correspondents in the field. By reviewing the notes,
        you write concise, descriptions for new articles. For example, if you are a category
        editor for the arts section and you read correspondent notes about street art in Asia,
        you generate a description like "street art in asia like graffiti and murals."
        Article descriptions should be specific enough to guide writers, not something broad like "art in asia."
        Article descriptions should not use specific references to people, organizations, locations, etc. 
        Article descriptions should have these characteristics:
        1. 10-20 words
        2. relate to the category name
        3. relate to the note provided.
        Provide the article description as a JSON with a key 'article_desc' for the description
        and a key 'kws' for a list of keywords to help writers find more information (also based on the note).
        No preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the correspondent note: \n\n {note} \n\n
        Here is the category name: {category} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    if(remove_entities):
        pa_prompt = pa_prompt1
    else:
        pa_prompt = pa_prompt0

    prompt_article_prompter = PromptTemplate(
        template=pa_prompt,
        input_variables=["note", "category"],
    )

    article_prompter = prompt_article_prompter | llm_json_out | JsonOutputParser()
    article_prompter = article_prompter.with_retry()
    
    return(article_prompter)

def get_article_grader(llm_model = 'llama3.1'):

    is_lmstudio_model = lmstudio_is_serving(llm_model)

    if(is_lmstudio_model is True):

        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # LM Studio's default API endpoint
            api_key="lmstudio",  # Dummy key; LM Studio doesn't require auth
            model=llm_model,
            temperature=0,
            timeout=10
        )

        prompt_article_grader = PromptTemplate(
            template="""You are a grader assessing relevance 
            of a document to a category. If the document is related to the category, grade it as relevant. 
            It does not need to be a stringent test. 
            The goal is to filter out erroneous or incoherent documents. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document 
            is coherent and relevant to the category. \n
            Provide the binary score as a JSON with a single key 'is_relevant' and no preamble or explanation.
            Here is the retrieved document: \n\n {article} \n\n
            Here is the category name: {category}
            """,
            input_variables=["article", "category"],
        )

    else:

        llm_json_out = ChatOllama(model=llm_model, format="json", 
            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

        prompt_article_grader = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are a grader assessing relevance 
            of a document to a category. If the document is related to the category, grade it as relevant. 
            It does not need to be a stringent test. 
            The goal is to filter out erroneous or incoherent documents. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document 
            is coherent and relevant to the category. \n
            Provide the binary score as a JSON with a single key 'is_relevant' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {article} \n\n
            Here is the category name: {category} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["article", "category"],
        )

    article_grader = prompt_article_grader | llm_json_out | JsonOutputParser()
    article_grader = article_grader.with_retry()
    
    return(article_grader)

def get_article_extension_grader(llm_model = 'llama3.1'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_extension_grader = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing relevance 
        of an answer to a question. If the content of the answer 
        1. Is responsive to the question and
        2. Offers data or detail in the answer such as names of people/places or organizations
        grade it as relevant. Otherwise, do not grade it as relevant.
        This should be a stringent test. 
        The goal is to filter out erroneous or overly broad, undetailed documents. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document 
        is coherent, detailed and relevant to the question. \n
        Provide the binary score as a JSON with a single key 'is_relevant' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the  document: \n\n {answer_content} \n\n
        Here is the question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["answer_content", "question"],
    )

    extension_grader = prompt_extension_grader | llm_json_out | JsonOutputParser()
    extension_grader = extension_grader.with_retry()
    
    return(extension_grader)

def get_group_meta_filterer(llm_model = 'llama3.1'):
    
    llm = ChatOllama(
        model=llm_model, 
        temperature=0, 
        stop=["<|eot_id|>"], timeout=10, num_predict= 3000
    )

    prompt_meta_filterer = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an editor. \n
        You edit english language articles by removing irrelevant parts of article content 
        which obviously do not relate to the article title. You do this by evaluating an article title 
        and then edit the article content
        so that it concisely relates to the title and remove any parts of the article content which do not
        relate to the title.
        For example, if you are given an article title 'New Developments in Chinese Businesses' and article
        content that has portions which relate to businesses in the US or Japan, you remove portions that
        relate to other countries from the article content and adjust article transitions 
        as necessary to keep the article content readable.
        Only remove obviously irrelevant portions; do not remove anything that is potentially 
        relevant to the title.
        Keep as much detail as possible, do not remove details if they are relevant to the article title.
        It is fine to edit for readability.
        No preamble or explanation or notes. Do not include an article title. 
        Only include the edited article content. \n 
        Do not repeat the question. \n
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Article Title: {articleTitle} \n\n 
        Article Content: {articleContent} \n\n
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["articleTitle","articleContent",],
    )

    group_meta_filterer = prompt_meta_filterer | llm | StrOutputParser()
    group_meta_filterer = group_meta_filterer.with_retry()
    
    return(group_meta_filterer)

def get_article_extender(llm_model = 'llama3.1'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_article_extender = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an editor. You review articles that are written by your staff and you generate
        a list of follow-up questions so that readers can learn more about aspects of the article.
        For example, if an article discusses specific people, companies, locations or subject matters,
        a reader might want to ask follow-up questions about those concepts. Specifically,
        if an article is about new legislation in China about education, some questions might relate
        to other legislation in China or about education legislation in other countries.
        Please keep the number of follow-up questions to 5 or less.  \n
        The question is standalone, therefore all relevant information in the statement of the question and
        there should be no terms like 'these developments' or 'this subject' because supporting information
        should be explicitly included in the statement of the question.
        Include specific information in the follow-up questions such as person names, company names, \n
        location names and/or subject matter from the article. \n
        No preamble or explanation. Do not include an article title. \n 
        Provide the follow-up as a JSON with a key
        'article_extensions' and the value as a list of questions.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Content: {context} \n\n 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["content"],
    )

    article_extender = prompt_article_extender | llm_json_out | JsonOutputParser()
    article_extender = article_extender.with_retry()
    
    return(article_extender)

def get_sec_headliner(llm_model = 'llama3.1'):
    
    llm_json_out = ChatOllama(model=llm_model, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a editor of a news aggregator website who writes engaging
         descriptions based on recent news articles; in order to write
        descriptions you should consider newsworthiness, relevance and general reader interest. 
        You are provided with the following information: 
        (1) the name of the overall subject matter of the paper,
        (2) the name of the specific section of interest.
        (3) summaries of potential candidates for the headline article.
        For instance, if the name of the overall subject matter is 'Technology' and the specific section
        of interest is 'Artificial Intelligence', you review the summaries of potential candidates,
        identify the best stories to highlight in the description
        based on newsworthiness, relevance and general reader
        interest and finally write a brief description, using the context provided.
        If there is not enough context information to answer the question, 
        respond with 'not enough information'. Provide the english language answer 
        in a JSON with a single key 'headline_story' and no preamble or explanation.
        Do not mention news or news articles. Description should be maximum four full sentences, formal,
        and interesting. Stay focused on a formal (but engaging) description. Do not include only titles.
        Provide specific information like names, organizations, people, etc. to get the reader interested.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the name of the overall subject matter: \n\n {main_subject} \n\n
        Here is the name of the section: \n\n {section_name} \n\n
        Here is a sample of recent articles: \n\n {recent_sample}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_prompt_section_headline = PromptTemplate(
        template=template_str,
        input_variables=['main_subject','section_name','recent_sample'],
    )

    sec_headliner = llm_prompt_section_headline | llm_json_out | JsonOutputParser()
    sec_headliner = sec_headliner.with_retry()

    return(sec_headliner)

def get_group_titler(llm_model = 'llama3.1'):
    
    llm = ChatOllama(
        model=llm_model, 
        temperature=0, 
        stop=["<|eot_id|>"], timeout=10, num_predict= 3000
    )
    
    group_title_analyst = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an objective editor. You write short, specific, english language titles 
        in 6 words or less by 
        (1) reading the content of an article, \n
        (2) considering the high-level topic of the article and \n
        (3) the general category of the article. \n
        For example, if an article is about 'trade' in the context of 'international relations' generally,
        then you would write a title that is relevant to that general space of articles.
        No preamble or explanation. Do not repeat the question. \n
        Include specifics like business/location names, product names, etc. if applicable. \n
        Do not try to make the title extra engaging or use casual words, stay concise and succinct. \n
        Use somewhat uncommon words.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context} 
        Topic: {childTopic} \n\n
        Category: {parentTopic} \n\n
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["context","childTopic","parentTopic"],
    )

    group_titler = group_title_analyst | llm | StrOutputParser()
    group_titler = group_titler.with_retry()
    
    return(group_titler)

def get_group_summarizer(llm_model = 'llama3.1', max_sentences=10, include_out = True):

    llm = ChatOllama(
        model=llm_model, 
        temperature=0, 
        stop=["<|eot_id|>"], timeout=10, num_predict= 3000
    )

    group_summary_analyst = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an analyst. \n
        You write short english language articles by reading headlines and excerpts and writing
        structured articles about a particular topic in the context of a general category. 
        For example, if you are writing an article
        about 'trade' in the context of 'international relations' generally and you are given articles about
        trade wars or export controls between countries, you produce an article that briefly describes the 
        relevance of trade in terms of international relations and then go into specifics based on 
        some or all of the provided articles.
        Please keep the article to less than """ + str(int(max_sentences)) + """ sentences.  \n
        Include specific, supporting information such as person names, company names, \n
        location names and/or examples from the context. \n
        In the beginning of the article, briefly position the topic before going into specifics. \n
        If you are writing about multiple stories, include transitions between the stories. \n
        Make sure the article is coherent and clearly structured, not a compilation of different facts. \n
        Try to include facts that are in multiple articles and not a single article. \n
        Do not include general information. \n
        No preamble or explanation. Do not include an article title. \n 
        Do not repeat the question. \n """ + include_out * """
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Context: {context} \n\n 
        Topic: {childTopic} \n\n
        Category: {parentTopic} \n\n
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["context","childTopic","parentTopic"],
    )

    group_summarizer = group_summary_analyst | llm | StrOutputParser()
    group_summarizer = group_summarizer.with_retry()
    
    return(group_summarizer)


def get_desc_generalizer(llm_model = 'llama3.2'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_generalizer = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an analyst
        who generalizes English language descriptions by simplifying them. \n
        You do this by adjusting descriptions to be more inclusive 
        while maintaining the main idea of the description provided.
        For example, if a description relates to red pants and shorts, updating could refer to pants broadly
        and give examples of pants that relate to the description provided.
        Provide the updated description as a JSON with a single key 'updated_description' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the description: \n\n {description} \n\n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["description"],
    )

    desc_generalizer = prompt_generalizer | llm_json_out | JsonOutputParser()
    desc_generalizer = desc_generalizer.with_retry()
    
    return(desc_generalizer)


def get_grader(llm_model='llama-3.2-3b-instruct', strict=False):
    is_lmstudio_model = lmstudio_is_serving(llm_model)

    if is_lmstudio_model:
        llm_json_out = ChatOpenAI(
            base_url=lmstudio_base_url,
            api_key="lmstudio",
            model=llm_model,
            temperature=0,
            timeout=10
        )

        # Define system and user prompts separately
        system_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question. "
            "Your goal is to filter out erroneous retrievals and give a binary score 'yes' or 'no' in JSON format. "
        ) + ("It should be a strict test." if strict else "")

        user_prompt = (
            "Here is the retrieved document:\n\n{document}\n\n"
            "Here is the user question: {question}\n"
            "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."
        )

        prompt_retrieval_grader = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_prompt)
        ])

    else:
        llm_json_out = ChatOllama(
            model=llm_model,
            format="json",
            temperature=0,
            stop=["<|eot_id|>"],
            timeout=10,
            num_predict=3000
        )

        prompt_retrieval_grader = PromptTemplate(
            template=(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|> "
                "You are a grader assessing relevance of a retrieved document to a user question. "
                "If the document is relevant to the user question, grade it as relevant. "
            ) + ("It should be a strict test. " if strict else "") + (
                "The goal is to filter out erroneous retrievals.\n"
                "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n"
                "Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n"
                "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                "Here is the retrieved document:\n\n{document}\n\n"
                "Here is the user question: {question}\n"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            ),
            input_variables=["question", "document"]
        )

    retrieval_grader = prompt_retrieval_grader | llm_json_out | JsonOutputParser()
    retrieval_grader = retrieval_grader.with_retry()

    return retrieval_grader


def get_summarizer(local_llm = 'llama-3.2-3b-instruct', local_timeout=20, is_short = True, include_out = True):

    is_lmstudio_model = lmstudio_is_serving(local_llm)

    if(is_lmstudio_model is False):

        llm = ChatOllama(
            model=local_llm, 
            temperature=0, 
            stop=["<|eot_id|>"], 
            timeout=local_timeout, 
            num_predict= 3000
        )

        summary_analyst = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are an analyst. \n
            You write english language summaries by reading headlines and excerpts. 
            Please write a """ + (is_short==True) * 'short,' + (is_short==False) * '' + """ 
            english summary \n
            Include specific, supporting information such as person names, company names, \n
            location names and examples from the context only. \n
            Keep the summary in paragraph form. \n
            Do not include general information. \n
            Do not mention people, organizations or places that are not in the context.
            No preamble or explanation. Do not repeat the question. \n """ + include_out * """
            If not an informative news article, output 'Not News Article'""" + """
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["context"],
        )
        
    else:

        llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # LM Studio's default API endpoint
            api_key="lmstudio",  # Dummy key; LM Studio doesn't require auth
            model=local_llm,
            temperature=0,
            timeout=local_timeout
        )

        summary_analyst = PromptTemplate(
            template="""
            You are an analyst. \n
            You write english language summaries by reading headlines and excerpts. 
            Please write a """ + (is_short) * 'short,' + (not is_short) * ' ' + """ 
            english summary \n
            Include specific, supporting information such as person names, company names, \n
            location names and examples from the context only. \n
            Keep the summary in paragraph form. \n
            Do not include general information. \n
            Do not mention people, organizations or places that are not in the context.
            No preamble or explanation. Do not repeat the question. \n """ + include_out * """
            If not an informative news article, output 'Not News Article'""" + """
            Context: {context} 
            """,
            input_variables=["context"],
        )

    

    summarizer = summary_analyst | llm | StrOutputParser()
    summarizer = summarizer.with_retry()
    
    return TimeoutRunnable(summarizer, timeout=local_timeout)


def get_gsm_identifier(local_llm='llama3.1', seedNum = 1):

    llm_json_out = ChatOllama(
        model=local_llm, 
        format="json", 
        temperature=0, 
        stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000, 
        seed = seedNum
        )

    key_feature_identifier = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an analyst 
        who categorizes documents; and you do this by identifying the key
        subcategories of documents. For instance a travel document may be subcategorized by
        a location and a desirable feature of that location or a news article may be identified by a
        specific person and a specific action by him/her. 
        You are provided with english language titles and excerpts from groups of documents,
        your goal is to identify at most 5 specific subcategories of subject matter from these titles.
        Order the subcategories from general subjects to specific ones.
        Give a few words to indicate subject matter, based on the titles provided, be as specific as possible. 
        Provide this response as a JSON with a single key 'subject_matter' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the document content: \n\n {doc_info} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["doc_info"],
    )

    group_sm_identifier = key_feature_identifier | llm_json_out | JsonOutputParser()
    group_sm_identifier = group_sm_identifier.with_retry()

    return(group_sm_identifier)


def get_prompt_expander(llm_model = 'llama3.1'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
                              temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_expander = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an researcher
        helping to expand questions by adding additional context that would be 
        helpful for retrieving information to provide an informed response to the prompt. 
        If the prompt does not include synonyms or keywords to help with retrieval, add the additional context. \n
        Provide the updated prompt as a JSON with a single key 'expanded_question' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the original prompt: \n\n {question} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    expander = prompt_expander | llm_json_out | JsonOutputParser()
    expander = expander.with_retry()
    
    return(expander)

def get_s_reorganizer():

    local_llm = 'llama3.1'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    content_reorganizer = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a sentence reorganizer 
        who orders english sentences into coherent paragraphs; and you do this by using directions from the system. 
        For instance sentences could be organized based on a list of relevant topics that are provided. 
        You are provided with a group of randomly ordered sentences and a sequential list of topics to reorganize the sentences,
        your goal is to reorganize the sentences provided only, based on the list of topics provided.
        Give a reordering of the sentences, based on the sequential list provided. 
        Provide this response as a JSON with a key for each topic and the associated sentences in a list, 
        and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the randomly ordered sentences: \n\n {sentences} \n\n
        Here is the order of topics to organize the sentences: \n\n {ordered_topics} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=['sentences','ordered_topics'],
    )

    sentence_reorganizer = content_reorganizer | llm_json_out | JsonOutputParser()
    sentence_reorganizer = sentence_reorganizer.with_retry()

    return(sentence_reorganizer)


def parse_title(rTitle):
    if rTitle[0] == '"':
        rTitle = [c for c in rTitle.split('"') if len(c)][0]
    elif '\n' in rTitle:
        rTitle = [c for c in rTitle.split('\n') if len(c)][0]
    if rTitle[-1] == ".":
        rTitle = rTitle[:-1]
    return rTitle


def get_titler(llm_model='llama-3.2-3b-instruct'):
    is_lmstudio_model = lmstudio_is_serving(llm_model)

    if is_lmstudio_model:
        llm = ChatOpenAI(
            base_url=lmstudio_base_url,
            api_key="lmstudio",
            model=llm_model,
            temperature=0,
            timeout=10
        )

        title_analyst = PromptTemplate(
            template="""
            You are an editor. You write short, engaging, specific, english language titles from summaries in 6 words or less. 
            Make sure to include detailed information such as person names, company names, location names and examples, etc. 
            Do not make a general title without specific names of people, organizations, locations, etc. 
            No preamble or explanation. Do not repeat the question. 
            Provide only one title.
            Context: {context} 
            """,
            input_variables=["context"],
        )
    else:
        llm = ChatOllama(
            model=llm_model,
            temperature=0,
            stop=["<|eot_id|>"],
            timeout=1,
            num_predict=3000
        )

        title_analyst = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are an editor. You write short, engaging, specific, english language titles from summaries in 6 words or less. 
            Make sure to include detailed information such as person names, company names, location names and examples, etc. 
            Do not make a general title without specific names of people, organizations, locations, etc. 
            No preamble or explanation. Do not repeat the question. 
            Provide only one title.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["context"],
        )

    # Compose the chain with parse_title as final step
    titler = title_analyst | llm | StrOutputParser() | RunnableLambda(parse_title)
    titler = titler.with_retry()

    return titler


def get_named_entity_0(llm_model = 'llama3.1'):
    llm_json_out = ChatOllama(
        model=llm_model, 
        format="json", 
        temperature=0, 
        stop=["<|eot_id|>"], 
        timeout=10, 
        num_predict= 3000
        )

    prompt_is_entity = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an analyst assessing if a user
        statement is correct based on the document provided. If the document shows that the statement is correct, 
        grade it as correct. It does not need to be a stringent test. The goal is to filter out erroneous statements. \n
        Give a binary score 'yes' or 'no' score to indicate whether the statement is correct. \n
        Provide the binary score as a JSON with a single key 'is_true' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user statement: {statement} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["statement", "document"],
    )

    named_entity_0 = prompt_is_entity | llm_json_out | JsonOutputParser()
    named_entity_0 = named_entity_0.with_retry()
    return(named_entity_0)

def get_named_entity_1(llm_model = 'llama3.1'):

    llm_json_out = ChatOllama(model=llm_model, format="json", 
                                temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    prompt_entity_type = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for 
        question-answering tasks. Use the document provided context to answer the question. \n
        If the document does not have enough information to answer the question, \n
        respond with 'not enough information'. The answer does not need to be perfect. \n
        The goal is to provide reasonable answers if there is enough information. \n
        Give the answer in 3 words or less or 'not enough information' to indicate there isn't enough information. \n
        Provide the answer as a JSON with a single key 'answer' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    named_entity_1 = prompt_entity_type | llm_json_out | JsonOutputParser()
    named_entity_1 = named_entity_1.with_retry()

    return(named_entity_1)

def get_taxon_overlap_grader(ntype):

    local_llm = 'llama3.1'
    
    ntype_context_dict = {
        'organization':{
            'name':'organization',
            'name_plural':'organizations',
            'same_0':'Huawei',
            'same_1':'Huawei Cloud',
            'diff_0':'Ministry of Commerce',
            'diff_1':'Ministry of Defense',
        },
        'person':{
            'name':'person',
            'name_plural':'people',
            'same_0':'President Biden',
            'same_1':'Joe Biden',
            'diff_0':'Mohammad Ali',
            'diff_1':'Mohammad Ali Rashed Lootah',
        },
        'location':{
            'name':'location',
            'name_plural':'locations',
            'same_0':'Atlantic',
            'same_1':'Atlantic Ocean',
            'diff_0':'America',
            'diff_1':'South America',
        },
    
    }

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                        temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)
    
    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a senior analyst who reviews taxonomies for errors. The current taxonomy contains
        a list of different """ + ntype_context_dict[ntype]['name_plural'] + """ and your job is to verify that values actually 
        represent different """ + ntype_context_dict[ntype]['name_plural'] + """ or if the values should be merged. 
        You are provided with a list of the values to consider and context about each value. 
        For instance, '""" + ntype_context_dict[ntype]['same_0'] + """' 
        and '""" + ntype_context_dict[ntype]['same_1'] + """' refer to the same general 
        """ + ntype_context_dict[ntype]['name'] + """; 
        while '""" + ntype_context_dict[ntype]['diff_0'] + """' 
        and '""" + ntype_context_dict[ntype]['diff_1'] + """' do not refer to the 
        same general """ + ntype_context_dict[ntype]['name'] + """ because they represent 
        distinct """ + ntype_context_dict[ntype]['name_plural'] + """. 
        Give a binary score of 'yes' or 'no' to indicate whether the values refer to 
        the same """ + ntype_context_dict[ntype]['name'] + """.
        This does not have to be a stringent test.
        Provide the binary score as a JSON with a single key 'is_same' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the values: \n\n {vals} \n\n
        Here is the context for those values: \n\n {val_context} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    taxon_diff = PromptTemplate(
        template=template_str,
        input_variables=['vals','val_context'],
    )

    taxon_grader = taxon_diff | llm_json_out | JsonOutputParser()
    taxon_grader = taxon_grader.with_retry()

    return(taxon_grader)


def get_taxon_eng_grader(ntype):

    llm_model = 'llama3.1'

    ntype_context_dict = {
        'organization':{
            'name':'organization',
            'name_plural':'organizations',
            'broad_0':'Foundry',
            'broad_1':'Apple',
        },
        'person':{
            'name':'person',
            'name_plural':'people',
            'broad_0':'Anthony',
            'broad_1':'Biden',
        },
        'location':{
            'name':'location',
            'name_plural':'locations',
            'broad_0':'West',
            'broad_1':'China',
        },

    }

    llm_json_out = ChatOllama(model=llm_model, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a senior analyst who reviews taxonomies for errors. The current taxonomy contains
        a list of different """ + ntype_context_dict[ntype]['name_plural'] + """ and your job is to identify 
        """ + ntype_context_dict[ntype]['name_plural'] + """ which are overly broad and should be removed. 
        You are provided with a specific """ + ntype_context_dict[ntype]['name'] + """ value to 
        consider and context about this value. 
        For instance, '""" + ntype_context_dict[ntype]['broad_0'] + """' 
        is too broad to represent a specific """ + ntype_context_dict[ntype]['name'] + """; 
        on the other hand, '""" + ntype_context_dict[ntype]['broad_1'] + """' actually refers to 
        a specific """ + ntype_context_dict[ntype]['name'] + """. 
        Give a binary score of 'yes' or 'no' to indicate whether the value refers to 
        an overly broad """ + ntype_context_dict[ntype]['name'] + """.
        If the provided context references the different """ + ntype_context_dict[ntype]['name_plural'] + """ 
        then respond with 'yes' otherwise respond with 'no'.
        Provide the binary score as a JSON with a single key 'is_broad' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the value: \n\n {val} \n\n
        Here is the context for the value: \n\n {val_context} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    taxon_eng_gen = PromptTemplate(
        template=template_str,
        input_variables=['val','val_context'],
    )

    taxon_eng_grader = taxon_eng_gen | llm_json_out | JsonOutputParser()
    taxon_eng_grader = taxon_eng_grader.with_retry()
    
    return(taxon_eng_grader)

def get_taxon_meta_org_ps(local_llm='llama3.1'):

    local_llm = 'llama3.1'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an analyst who answers questions. Your job is to describe 
        the products and/or services that are provided by an organization. 
        You are provided with a specific context about this organization. 
        The goal is to provide a succinct list of products and/or services 
        only if there is enough information in the context provided.
        If there is enough information to describe multiple products and/or services, it is okay to list
        multiple products and/or services. If there is not enough context information to answer the question, 
        respond with 'not enough information' or an empty list. Provide the english language answer 
        as a list in a JSON with a single key 'products_services' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the organization: \n\n {val} \n\n
        Here is the context for the organization: \n\n {val_context} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    taxon_meta_ps_prompt = PromptTemplate(
        template=template_str,
        input_variables=['val','val_context'],
    )

    taxon_meta_ps = taxon_meta_ps_prompt | llm_json_out | JsonOutputParser()
    taxon_meta_ps = taxon_meta_ps.with_retry()
    
    return(taxon_meta_ps)

def get_taxon_meta_org(local_llm='llama3.1'):

    local_llm = 'llama3.1'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an analyst who answers questions. Your job is to describe 
        organizations based on article information. 
        You are provided with a specific context about this organization. 
        The goal is to provide a short, succinct description of the organization 
        only if there is enough information in the context provided.
        If there is not enough context information to answer the question, 
        respond with 'not enough information' or an empty list. Provide the english language answer 
        as a string in a JSON with a single key 'org_desc' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the organization: \n\n {val} \n\n
        Here is the context for the organization: \n\n {val_context} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    taxon_meta_ps_prompt = PromptTemplate(
        template=template_str,
        input_variables=['val','val_context'],
    )

    taxon_meta_ps = taxon_meta_ps_prompt | llm_json_out | JsonOutputParser()
    taxon_meta_ps = taxon_meta_ps.with_retry()
    
    return(taxon_meta_ps)

def get_ner_meta(ner_meta_dict, local_llm='llama3.2'):

    local_llm = 'llama3.2'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a senior taxonomy analyst. 
        Your job is to categorize a """ + ner_meta_dict['ner_type'] + """ 
        according to """ + ner_meta_dict['meta_type'] + """. 
        You are provided with a specific context about this """ + ner_meta_dict['ner_type'] + """. 
        The goal is to provide a succinct """ + ner_meta_dict['meta_type'] + """ with 
        """ + ner_meta_dict['meta_length'] + """.
        For example if """ + ner_meta_dict['meta_example'] + """.
        If there is not enough context information to answer the question, 
        respond with 'not enough information'. Provide the english language answer 
        as a JSON with a single key '""" + ner_meta_dict['meta_type'] + """' 
        and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the """ + ner_meta_dict['ner_type'] + """: \n\n {val} \n\n
        Here is the context for the """ + ner_meta_dict['ner_type'] + """: \n\n {val_context} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    taxon_meta_ps_prompt = PromptTemplate(
        template=template_str,
        input_variables=['val','val_context'],
    )

    taxon_meta_ps = taxon_meta_ps_prompt | llm_json_out | JsonOutputParser()
    taxon_meta_ps = taxon_meta_ps.with_retry()
    
    return(taxon_meta_ps)


def get_is_reasonabler(local_llm = 'llama-3.1-8b-instruct', is_strict=False, local_timeout=10):

    is_lmstudio_model = lmstudio_is_serving(local_llm)

    if(is_lmstudio_model is False):

        llm_json_out = ChatOllama(
            model=local_llm, 
            format="json", 
            temperature=0, 
            stop=["<|eot_id|>"], 
            timeout=10, 
            num_predict= 3000)

        prompt_is_reasonable = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            You are an analyst assessing if a user
            statement is """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """ 
            based on the document provided. 
            If the document shows that the statement is """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """, 
            grade it as """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """. 
            """ + (is_strict==False)*'It does not need to be a stringent test. ' + (is_strict==True)*' ' + """
            Give a binary score 'yes' or 'no' score to indicate whether the statement is 
            """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """ . \n
            Provide the binary score as a JSON with a single key 'is_reasonable' and no preamble or explanation.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user statement: {statement} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["statement", "document"],
        )
        
    else:

        llm_json_out = ChatOpenAI(
            base_url="http://localhost:1234/v1",  # LM Studio's default API endpoint
            api_key="lmstudio",  # Dummy key; LM Studio doesn't require auth
            model=local_llm,
            temperature=0,
            timeout=local_timeout
        )

        template_str = """
            You are an analyst assessing if a user
            statement is """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """ 
            based on the document provided. 
            If the document shows that the statement is """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """, 
            grade it as """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """. 
            """ + (is_strict==False)*'It does not need to be a stringent test. ' + (is_strict==True)*' ' + """
            Give a binary score 'yes' or 'no' score to indicate whether the statement is 
            """ + (is_strict==False)*'reasonable' + (is_strict==True)*'correct' + """ . \n
            Provide the binary score as a JSON with a single key 'is_reasonable' and no preamble or explanation.
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user statement: {statement} \n
            """

        prompt_is_reasonable = PromptTemplate(
            template=template_str,
            input_variables=["statement", "document"],
        )

    
    gen_q_func = prompt_is_reasonable | llm_json_out | JsonOutputParser()
    gen_q_func = gen_q_func.with_retry()

    
    return TimeoutRunnable(gen_q_func, timeout=local_timeout)

def get_tree_is_subcat():
    local_llm = 'llama3.1'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a senior taxonomy analyst who reviews groups of news articles and verifies the
        name of the subcategory. You are provided with the following information: 
        (1) a list of news articles that should relate to the proposed name, 
        (2) the parent category name.
        (3) the proposed name for the subcategory of articles that the news articles come from.
        For instance, if the name of the parent category is 'Sports' and the news article
        titles generally describe sports items like rackets/balls/etc., and the proposed name is 
        "equipment" then that proposed name is reasonable. 
        Subcategory names should not include the name of the parent category.
        If there is not enough context information to answer the question, 
        respond with 'not enough information' or an empty list. Provide the english language answer 
        in a JSON with a single key 'is_subcategory' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the news articles: \n\n {article_content} \n\n
        Here is the name of the parent category: \n\n {topic_name} \n\n
        Here is the proposed subcategory name: \n\n {subcat_name} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_ques_prompt = PromptTemplate(
        template=template_str,
        input_variables=['article_content','subcat_name','topic_name'],
    )

    is_subcater = llm_ques_prompt | llm_json_out | JsonOutputParser()
    is_subcater = is_subcater.with_retry()
    return(is_subcater)

def get_tree_subcat_descer():

    local_llm = 'llama3.1'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a senior taxonomy analyst who writes brief subcategory descriptions. 
        You are provided with the following information: 
        (1) a sample of news articles,
        (2) the name of the subcategory of the taxonomy.
        For instance, if the name of the subcategory name is 'Sports' then you would 
        succinctly write a few sentences describing this subcategory 
        using the context provided.
        If there is not enough context information to answer the question, 
        respond with 'not enough information' or an empty list. Provide the english language answer 
        in a JSON with a single key 'subcategory_description' and no preamble or explanation.
        Do not mention news or news articles. Description should be succinct, related to the subcategory topic 
        and only rely on the context provided.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the articles: \n\n {article_content} \n\n
        Here is the name of news article subcategory: \n\n {subcat_name} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_ques_prompt = PromptTemplate(
        template=template_str,
        input_variables=['article_content','subcat_name'],
    )

    subcategory_descer = llm_ques_prompt | llm_json_out | JsonOutputParser()
    subcategory_descer = subcategory_descer.with_retry()
    
    return(subcategory_descer)

def get_tree_subcat_correct():

    local_llm = 'llama3.1'

    llm_json_out = ChatOllama(model=local_llm, format="json", 
                            temperature=0, stop=["<|eot_id|>"], timeout=10, num_predict= 3000)

    template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a senior taxonomy analyst who reviews if an article has been properly assigned 
        to a defined subcategory of topics. You are provided with the following information: 
        (1) article content 
        (2) name and description of the assigned subcategory
        For instance, if an article is primarily describing economic trends and the assigned
        subcategory relates to technologies then that proposed name is not correct. 
        This evaluation does not need to be a strict test, the goal is to identify obvious mistakes.
        Provide the english language answer 
        in a JSON with a single key 'is_correct' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the text of the article: \n\n {article_content} \n\n
        Here is the name and description of the subcategory: \n\n {subcat_desc} \n\n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

    llm_ques_prompt = PromptTemplate(
        template=template_str,
        input_variables=['article_content','subcat_desc'],
    )

    is_subcat_righter = llm_ques_prompt | llm_json_out | JsonOutputParser()
    is_subcat_righter = is_subcat_righter.with_retry()

    return(is_subcat_righter)