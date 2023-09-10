
import metaphor_python
import os

# Initialize the Metaphor API client with your API key
metaphor_client = metaphor_python.Metaphor(api_key="536446c2-3819-46ad-891e-21ff1d67ccf3")

def find_links_for_question(summary):
    search_response = metaphor_client.search("Here is a paragraph of information:" + summary + "Here are some related links:", num_results=3)
    contents_response = search_response.get_contents()
    # Print content for each result
    links = []
    for content in contents_response.contents:
        print(f"URL: {content.url}\n")
        links.append(content.url)
    return links

sample="This week, the five main reasons why people lose at poker are discussed. Bad luck is number five on the list, but if bad luck has been going on for more than a few months, it's time to look at other factors such as predictability, playing tough games, lack of fundamentals, and tilt. With practice and awareness, these issues can be addressed and poker skills improved"
find_links_for_question(sample)

# sample = " Number one reason you may be losing at poker is tilt. To dispel tilt, one should quit earlier than they think they should and try to become present to the situation. To become present, one should think about their thoughts, feelings, and physical sensations. When present, one should choose how they want to be - relaxed, calm, and focused."
# find_links_for_question(sample)