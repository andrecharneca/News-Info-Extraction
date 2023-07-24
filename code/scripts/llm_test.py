from time import time
import os
import json
import sys
from pprint import pprint
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
sys.path.append("../src/")
from pipelines.paths import ARTICLES_JSONS_PATH

prompt = """Each article below has an associated list of entities and a table with relationships. The list has items of the form "entity (type)", where "entity" is the name (e.g. Amazon) and "type" is one of these types: ORG (for companies) or PROD (for products). The table links the mentioned companies to other mentioned companies. Here's the explanation of every column. "Entity" relates to a company or a product. "Relationship" is the connection between the entities, only using entities from the list. For example, in the sentence "KitKat, produced by Nestle, is working with a new cocoa farm", a valid relationship is |Nestle|developer of|KitKat|, but a non-valid relationship is |KitKat|partner of|new cocoa farm|, because "new cocoa farm" is not a valid entity. "Passage" is the text from the article relevant to that relationship. Some articles don't have entities (Entity list = []) or no relationships (the only table row is |end|).
Allowed relationship types:
Company-Company (ORG-ORG) = [owner of, partner of, investor in, competitor]
Company-Product (ORG-PROD) = [developer of]
Product-Product (PROD-PROD) = [competitor]
Here are some examples:
Title: JPMorgan restricts employee use of ChatGPT
Date: Wed February 22, 2023
Text: JPMorgan Chase is temporarily clamping down on the use of ChatGPT among its employees, as the buzzy AI chatbot explodes in popularity.
The biggest US bank has restricted its use among global staff, according to a person familiar with the matter. The decision was taken not because of a particular issue, but to accord with limits on third-party software due to compliance concerns, the person said. JPMorgan Chase (JPM) declined to comment.
ChatGPT was released to the public in late November by artificial intelligence research company Open AI. Since then, the much-hyped tool has been used to turn written prompts into convincing academic essays and creative scripts as well as trip itineraries and computer code.
Adoption has skyrocketed. UBS estimated that ChatGPT reached 100 million monthly active users in January, two months after its launch. That would make it the fastest-growing online application in history, according to the Swiss bank’s analysts.
The viral success of ChatGPT has kickstarted a frantic competition among tech companies to rush AI products to market. Google recently unveiled its ChatGPT competitor, which it’s calling Bard, while Microsoft (MSFT), an investor in Open AI, debuted its Bing AI chatbot to a limited pool of testers.
But the releases have boosted concerns about the technology. Demos of both Google and Microsoft’s tools have been called out for producing factual errors. Microsoft, meanwhile, is trying to rein in its Bing chatbot after users reported troubling responses, including confrontational remarks and dark fantasies. Another big player in AI innovation is Meta, with a higher focus on open source models. Their recent push for the Metaverse and the Oculus glasses has put them behind the chatbot race.
Some businesses have encouraged workers to incorporate ChatGPT into their daily work. But others worry about the risks. The banking sector, which deals with sensitive client information and is closely watched by government regulators, has extra incentive to tread carefully.
Schools are also restricting ChatGPT due to concerns it could be used to cheat on assignments. New York City public schools banned it in January.
Entity list = [JPMorgan (ORG), ChatGPT (PROD), Open AI (ORG), Google (ORG), Meta (ORG), Metaverse (PROD), Oculus (PROD), UBS (ORG), Bard (PROD), Microsoft (ORG)]
|Entity|Relationship|Entity|Passage|
|Open AI|developer of|ChatGPT|"ChatGPT was released to the public in late November by artificial intelligence research company Open AI"|
|Google|competitor|Open AI|"Google recently unveiled its ChatGPT competitor"|
|Microsoft|investor in|Open AI|"Microsoft (MSFT), an investor in Open AI"|
|Bard|competitor|ChatGPT|"Google recently unveiled its ChatGPT competitor, which it’s calling Bard"
|Google|competitor|Microsoft|"Demos of both Google and Microsoft’s tools"|
|Meta|competitor|Microsoft|"Another big player in AI innovation is Meta"|
|Meta|competitor|Google|"Another big player in AI innovation is Meta"|
|Meta|competitor|OpenAI|"Another big player in AI innovation is Meta"|
|Meta|developer of|Metaverse|"Their recent push for the Metaverse"|
|Meta|developer of|Oculus|"Their recent push for the Metaverse and the Oculus glasses"|
|end|
--
Title: Amplifon expects acquisitions to boost revenue after record full-year results
Date: March 1, 2023
Text: March 1 (Reuters) - Italy's Amplifon (AMPF.MI) expects to boost revenue through bolt-on acquisitions this year, the world's largest hearing aid retailer said on Wednesday after reporting record core profit for 2022.
The Milan-based company posted full-year recurring earnings before interest, taxes, depreciation, and amortisation (EBITDA) of 525.3 million euros ($560.28 million) in its best-ever results, compared with 482.8 million euros a year earlier.
Amplifon, which proposed a divided of 29 cents per share, also reported record annual recurring revenue of 2.12 billion euros, compared with 1.95 billion euros a year earlier.
By 1241 GMT, the company's shares were up 1.7%, while Italy's blue-chip index FTSE MIB (.FTMIB) was up 0.65%.
Entity list = [Amplifon (ORG)]
|Entity|Relationship|Entity|Passage|
|end|
--
Title: Tensions between gaming companies soar to new highs
Date: Fri, June 20, 2021
Text: Microsoft’s pending $68.7 billion acquisition of Activision-Blizzard has been a highly debated topic in recent times. The conversation around the deal only appears to be intensifying, particularly after Microsoft’s recent meeting with EU regulators. During the meeting, Microsoft looked to make its case, mostly countering Sony’s opposition to the deal. One of the biggest points about the Activision purchase pertains to the potential implications for the Call of Duty franchise.
A similar tension was observed back in 2016 with From Software, the studio behind the famous Dark Souls franchise. It was acquired by Bandai in a similar fashion, leading to high amounts of speculation. Bandai Namco, since last year, has partnered with Activision to publish their most important titles but has since decided to stop this venture, as a report from last week indicates. Some rumors have been circulating about Mojang being Activision's replacement for Bandai Namco's upcoming game releases, which had the markets going wild.
The latest player in the videogame industry, Meta (previously known as Facebook), is also looking to take advantage of its massive user base to enter the gaming industry, taking advantage of their VR technology.
Entity list = [Microsoft (ORG), Activision-Blizzard (ORG), Sony (ORG), Call of Duty (PROD), From Software (ORG), Dark Souls (PROD), Bandai Namco (ORG), Meta (ORG), Facebook (ORG)]
|Entity|Relationship|Entity|Passage|
|Microsoft|owner of|Activision-Blizzard|"Microsoft’s pending $68.7 billion acquisition of Activision-Blizzard"|
|Microsoft|competitor|Sony|"Microsoft looked to make its case, mostly countering Sony’s opposition to the deal"|
|Bandai Namco|owner of|From Software|"From Software, the studio behind of the famous Dark Souls franchise. It was acquired by Bandai in a similar fashion"|
|Bandai Namco|partner of|Activision|"Bandai Namco, since last year, has partnered with Activision"|
|Bandai Namco|partner of|Mojang|"Some rumors have been circulating about Mojang being Activision's replacement for Bandai Namco's upcoming game releases"|
|Meta|competitor|Microsoft|"The latest player in the videogame industry, Meta (previously known as Facebook), is also looking to take advantage of its massive user base to enter the gaming industry"|
|From Software|developer of|Dark Souls|"From Software, the studio behind the famous Dark Souls franchise"|
|Activision-Blizzard|developer of|Call of Duty|"Activision purchase pertains to the potential implications for the Call of Duty franchise"|
|end|
--
Title: Microsoft and NVIDIA announce expansive new gaming deal
Date:
Text: Partnership will bring blockbuster lineup of Xbox games, including Minecraft and Activision titles like Call of Duty, to NVIDIA GeForce NOW cloud gaming service
REDMOND, Wash. and SANTA CLARA, Calif., Feb. 21, 2023 /PRNewswire/ -- On Tuesday, Microsoft and NVIDIA announced the companies have agreed to a 10-year partnership to bring Xbox PC games to the NVIDIA® GeForce NOW™ cloud gaming service, which has more than 25 million members in over 100 countries.
The agreement will enable gamers to stream Xbox PC titles from GeForce NOW to PCs, macOS, Chromebooks, smartphones and other devices. It will also enable Activision Blizzard PC titles, such as Call of Duty, to be streamed on GeForce NOW after Microsoft's acquisition of Activision closes.
"Xbox remains committed to giving people more choice and finding ways to expand how people play," said Microsoft Gaming CEO Phil Spencer. "This partnership will help grow NVIDIA's catalog of titles to include games like Call of Duty, while giving developers more ways to offer streaming games. We are excited to offer gamers more ways to play the games they love."
"Combining the incredibly rich catalog of Xbox first party games with GeForce NOW's high-performance streaming capabilities will propel cloud gaming into a mainstream offering that appeals to gamers at all levels of interest and experience," said Jeff Fisher, senior vice president for GeForce at NVIDIA. "Through this partnership, more of the world's most popular titles will now be available from the cloud with just a click, playable by millions more gamers."
The partnership delivers increased choice to gamers and resolves NVIDIA's concerns with Microsoft's acquisition of Activision Blizzard. NVIDIA therefore is offering its full support for regulatory approval of the acquisition.
Microsoft and NVIDIA will begin work immediately to integrate Xbox PC games into GeForce NOW, so that GeForce NOW members can stream PC games they buy in the Windows Store, including third-party partner titles where the publisher has granted streaming rights to NVIDIA. Xbox PC games currently available in third-party stores like Steam or Epic Games Store will also be able to be streamed through GeForce NOW.
Visit the GeForce NOW website for more information on the service and follow along every GFN Thursday for the latest news, including release dates for upcoming Microsoft game titles coming to the GeForce NOW service.
The agreement was announced today at a Microsoft press conference in Brussels, Belgium. Microsoft also shared today that it finalized a 10-year agreement to bring the latest version of Call of Duty to the Nintendo platform following the merger with Activision.
About NVIDIA
Since its founding in 1993, NVIDIA (NASDAQ: NVDA) has been a pioneer in accelerated computing. The company's invention of the GPU in 1999 sparked the growth of the PC gaming market, redefined computer graphics, ignited the era of modern AI and is fueling the creation of the metaverse. NVIDIA is now a full-stack computing company with data-center-scale offerings that are reshaping industry. More information at https://nvidianews.nvidia.com/.
About Microsoft
Microsoft (Nasdaq "MSFT" @microsoft) enables digital transformation for the era of an intelligent cloud and an intelligent edge. Its mission is to empower every person and every organization on the planet to achieve more.
Certain statements in this press release including, but not limited to, statements as to: the benefits, impact, and performance of NVIDIA's products and technologies, including GeForce NOW; and NVIDIA's partnership with Microsoft and the benefits and impact there of are forward-looking statements that are subject to risks and uncertainties that could cause results to be materially different than expectations. Important factors that could cause actual results to differ materially include: global economic conditions; NVIDIA's reliance on third parties to manufacture, assemble, package and test its products; the impact of technological development and competition; development of new products and technologies or enhancements to NVIDIA's existing product and technologies; market acceptance of NVIDIA's products or NVIDIA's partners' products; design, manufacturing or software defects; changes in consumer preferences or demands; changes in industry standards and interfaces; unexpected loss of performance of NVIDIA's products or technologies when integrated into systems; as well as other factors detailed from time to time in the most recent reports NVIDIA files with the Securities and Exchange Commission, or SEC, including, but not limited to, its annual report on Form 10-K and quarterly reports on Form 10-Q. Copies of reports filed with the SEC are posted on NVIDIA's website and are available from NVIDIA without charge. These forward-looking statements are not guarantees of future performance and speak only as of the date hereof, and, except as required by law, NVIDIA disclaims any obligation to update these forward-looking statements to reflect future events or circumstances.
© 2023 NVIDIA Corporation. All rights reserved. NVIDIA, the NVIDIA logo, GeForce and GeForce NOW are trademarks and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries. Other company and product names may be trademarks of the respective companies with which they are associated. Features, pricing, availability and specifications are subject to change without notice.
SOURCE Microsoft Corp.
Entity list ="""

default_args = {
    "prompt": prompt,
    "model_name": "StabilityAI/stablelm-base-alpha-7b",
    "torch_dtype": torch.bfloat16,
    "max_new_tokens": 300,
    "temperature": 0.001,
    "do_sample": True,
    "num_return_sequences": 1,
    "load_in_8bit": False,
    "stream": True,
}

def load_article_id(article_id):
    article_file = os.path.join(ARTICLES_JSONS_PATH, f"article_id_{article_id}.json")
    with open(article_file, "r") as f:
        article = json.load(f)
    article_text = article["text"]
    article_title = article["title"]
    return article_title, article_text

def load_llm(**args):
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    pipeline = transformers.pipeline(
        "text-generation",
        model=args['model_name'],
        tokenizer=tokenizer,
        torch_dtype=args['torch_dtype'],
        trust_remote_code=True,
        device_map="auto",
        model_kwargs = {"load_in_8bit":args["load_in_8bit"]}
    )
    print("Device map = ", pipeline.model.hf_device_map)
    print(f"\t{args['model_name']} = {pipeline.model.get_memory_footprint()/1e9:.2f} GB of memory")
    return pipeline

def run_llm(pipeline, **args):
    start = time()
    streamer = TextStreamer(pipeline.tokenizer) if args['stream'] else None
    sequences=pipeline(
    args["prompt"],
        max_new_tokens=args['max_new_tokens'],
        temperature=args['temperature'],
        do_sample=args['do_sample'],
        num_return_sequences=args['num_return_sequences'],
        eos_token_id=pipeline.tokenizer.eos_token_id,
        return_full_text=False,
        streamer=streamer,
    )
    stop = time()
    for sequence in sequences:
        print(f"*** Response ***\n{sequence['generated_text']}\n***************\n")
        tokens = pipeline.tokenizer(sequence["generated_text"])
        print(f"Time = {(stop-start)/len(tokens['input_ids']):.2f} s/token\n")


def text_generation():
    args = default_args
    load_llm(**args)
    run_llm(**args)

def chatbot():
    args = default_args
    args['model_name'] = "StabilityAI/stablelm-tuned-alpha-7b"
    pipeline=load_llm(**args)

    # read json file with prompts
    with open("../../data/temp/chatbot_prompts.json", "r") as f:
        prompts = json.load(f)

    system_prompt = prompts["system_prompt"]
    user_prompt = prompts["user_prompt"]
    chat_prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"

    args['prompt'] = chat_prompt
    run_llm(pipeline,**args)


    


def main():
    chatbot()

if __name__ == "__main__":
    main()
    




