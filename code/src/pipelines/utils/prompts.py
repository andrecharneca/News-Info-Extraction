ENTITY_REL_ENTITY_PROMPT = """Each article below has an associated table. The table links the mentioned companies to other mentioned companies. Here's the explanation of every column. "Entity" relates to a company, an important business person or important product. "Relationship" is the connection between these entities shown in the article, and there's a limited set of possible relationships between entities: [owner of, partner of, developer of, investor in, competitor, rebranded as]. For example, in the sentence "KitKat, owned by Nestle, is working with a new cocoa farm", a valid relationship is |Nestle|owner of|KitKat|, but a non-valid relationship is |KitKat|partner of|new cocoa farm|, because "new cocoa farm" is not a valid entity."Date" is the time when that relationship happened, some examples are [2018, this Wednesday, future, now]. If there's not enough information in the article to determine this, use the value "null".  "Passage" is one or more pieces of text from the article relevant to that relationship. Here are some examples:
Title: JPMorgan restricts employee use of ChatGPT
Date: Wed February 22, 2023
Text: JPMorgan Chase is temporarily clamping down on the use of ChatGPT among its employees, as the buzzy AI chatbot explodes in popularity.
The biggest US bank has restricted its use among global staff, according to a person familiar with the matter. The decision was taken not because of a particular issue, but to accord with limits on third-party software due to compliance concerns, the person said. JPMorgan Chase (JPM) declined to comment.
ChatGPT was released to the public in late November by artificial intelligence research company Open AI. Since then, the much-hyped tool has been used to turn written prompts into convincing academic essays and creative scripts as well as trip itineraries and computer code.
Adoption has skyrocketed. UBS estimated that ChatGPT reached 100 million monthly active users in January, two months after its launch. That would make it the fastest-growing online application in history, according to the Swiss bank’s analysts.
The viral success of ChatGPT has kickstarted a frantic competition among tech companies to rush AI products to market. Google recently unveiled its ChatGPT competitor, which it’s calling Bard, while Microsoft (MSFT), an investor in Open AI, debuted its Bing AI chatbot to a limited pool of testers.
But the releases have boosted concerns about the technology. Demos of both Google and Microsoft’s tools have been called out for producing factual errors. Microsoft, meanwhile, is trying to rein in its Bing chatbot after users reported troubling responses, including confrontational remarks and dark fantasies.
Some businesses have encouraged workers to incorporate ChatGPT into their daily work. But others worry about the risks. The banking sector, which deals with sensitive client information and is closely watched by government regulators, has extra incentive to tread carefully.
Schools are also restricting ChatGPT due to concerns it could be used to cheat on assignments. New York City public schools banned it in January.
|Entity|Relationship|Entity|Passage|
|OpenAI|developer of|ChatGPT|"ChatGPT was released to the public in late November by artificial intelligence research company Open AI"|
|Google|competitor|OpenAI|"Google recently unveiled its ChatGPT competitor"|
|Microsoft|investor in|OpenAI|"Microsoft (MSFT), an investor in Open AI"|
|Bard|competitor|ChatGPT|"Google recently unveiled its ChatGPT competitor, which it’s calling Bard"
|Google|competitor|Microsoft|"Google recently unveiled its ChatGPT competitor"|
|end|
—-
Title: Tensions between gaming companies soar to new highs
Date: Fri, June 20, 2021
Text: Microsoft’s pending $68.7 billion acquisition of Activision-Blizzard has been a highly debated topic in recent times. The conversation around the deal only appears to be intensifying, particularly after Microsoft’s recent meeting with EU regulators. During the meeting, Microsoft looked to make its case, mostly countering Sony’s opposition to the deal. One of the biggest points about the Activision purchase pertains to the potential implications for the Call of Duty franchise.
A similar tension was observed back in 2016 with From Software, the studio behind the famous Dark Souls franchise. It was acquired by Bandai in a similar fashion, leading to high amounts of speculation. Bandai Namco, since last year, has partnered with Activision to publish their most important titles but has since decided to stop this venture, as a report from last week indicates. Some rumors have been circulating about Mojang being Activision's replacement for Bandai Namco's upcoming game releases, which had the markets going wild.
The latest player in the videogame industry, Meta (previously known as Facebook), is also looking to take advantage of its massive user base to enter the gaming industry, taking advantage of their VR technology.
|Entity|Relationship|Entity|Passage|
|Microsoft|owner of|Activision-Blizzard|"Microsoft’s pending $68.7 billion acquisition of Activision-Blizzard"|
|Microsoft|competitor|Sony|"Microsoft looked to make its case, mostly countering Sony’s opposition to the deal"|
|Bandai Namco|owner of|From Software|"From Software, the studio behind of the famous Dark Souls franchise. It was acquired by Bandai in a similar fashion"|
|Bandai Namco|partner of|Activision|"Bandai Namco, since last year, has partnered with Activision"|
|Bandai Namco|partner of|Mojang|"Some rumors have been circulating about Mojang being Activision's replacement for Bandai Namco's upcoming game releases"|
|Meta|competitor|Microsoft|"The latest player in the videogame industry, Meta (previously known as Facebook), is also looking to take advantage of its massive user base to enter the gaming industry"|
|Facebook|rebranded as|Meta|"Meta (previously known as Facebook)"|
|end|
—-
Title: Amplifon expects acquisitions to boost revenue after record full-year results
Date: March 1, 2023
Text: March 1 (Reuters) - Italy's Amplifon (AMPF.MI) expects to boost revenue through bolt-on acquisitions this year, the world's largest hearing aid retailer said on Wednesday after reporting record core profit for 2022.
The Milan-based company posted full-year recurring earnings before interest, taxes, depreciation, and amortisation (EBITDA) of 525.3 million euros ($560.28 million) in its best-ever results, compared with 482.8 million euros a year earlier.
Amplifon, which proposed a divided of 29 cents per share, also reported record annual recurring revenue of 2.12 billion euros, compared with 1.95 billion euros a year earlier.
By 1241 GMT, the company's shares were up 1.7%, while Italy's blue-chip index FTSE MIB (.FTMIB) was up 0.65%.
|Company|Relationship|Company|Passage|
|end|
—-
Title: Tanzania detects its first-ever cases of the highly fatal Marburg viral disease
Date: March 22, 2023
Text: DAR ES SALAAM, March 22 (Reuters) - Tanzania has confirmed its first-ever cases of Marburg, a high-fatality viral hemorrhagic fever with symptoms broadly similar to those of Ebola, the World Health Organisation (WHO) said.
The WHO said in a late Tuesday statement that the confirmation of the disease by Tanzania\'s national public laboratory followed the death of five of eight people in Tanzania\'s northwest Kagera region who developed symptoms, which include fever, vomiting, bleeding and renal failure.
Among the dead was a health worker, the WHO said. The three who survived were getting treatment, with 161 contacts being monitored.
"The efforts by Tanzania\'s health authorities to establish the cause of the disease is a clear indication of the determination to effectively respond to the outbreak," said Matshidiso Moeti, WHO regional director for Africa.
"We are working with the government to rapidly scale up control measures to halt the spread of the virus."With a fatality rate of as high as 88%, Marburg is from the same virus family responsible for Ebola and is transmitted to people from fruit bats. It then spreads through contact with bodily fluids of infected people.
Symptoms include high fever, severe headache and malaise which typically develop within seven days of infection, according to the WHO.
Equatorial Guinea is also battling its first-ever outbreak of Marburg that was confirmed in February.
Writing by Elias Biryabarema; Editing by Robert Birsel
Our Standards: The Thomson Reuters Trust Principles.
|Company|Relationship|Company|Passage|
|end|
--
Title: {title}
Date: {date}
Text: {text}
|Company|Relationship|Company|Passage|"""

ENTITY_REL_ENTITY_WITH_ENTITY_LIST_PROMPT = """Each article below has an associated list of entities and a table with relationships. The list has items of the form "entity (type)", where "entity" is the name (e.g. Amazon) and "type" is one of these types: ORG (for companies) or PROD (for products). The table links the mentioned companies to other mentioned companies. Here's the explanation of every column. "Entity" relates to a company or a product. "Relationship" is the connection between the entities, only using entities from the list. For example, in the sentence "KitKat, produced by Nestle, is working with a new cocoa farm", a valid relationship is |Nestle|developer of|KitKat|, but a non-valid relationship is |KitKat|partner of|new cocoa farm|, because "new cocoa farm" is not a valid entity. "Passage" is the text from the article relevant to that relationship. Some articles don't have entities (Entity list = []) or no relationships (the only table row is |end|).
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
|Bard|competitor|ChatGPT|"Google recently unveiled its ChatGPT competitor, which it’s calling Bard"|
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
Title: {title}
Date: {date}
Text: {text}
Entity list = """
