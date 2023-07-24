RELATIONSHIP_LIST = ['owner of', 'partner of', 'developer of', 'investor in', 'competitor']
RELATIONSHIP_TO_SENTENCE_DICT = {
        'owner of': "is the owner of", # Unidirectional: entity1 rel_sentence entity2
        'developer of': "is the developer of",
        'investor in': "is an investor in",
        'partner of': "are business partners", # Bidirectional: entity1 and entity2 rel_sentence
        'competitor': "are competitors",
    }

## Directionality ##
UNIDIR_RELATIONSHIP_LIST = ['owner of', 'developer of', 'investor in']
BIDIR_RELATIONSHIP_LIST = ['partner of', 'competitor']

UNIDIR_STRING_FORMAT = "{entity_1} {relationship} {entity_2}."
BIDIR_STRING_FORMAT = "{entity_1} and {entity_2} {relationship}."

## Entity types ##
ORG_ORG_RELATIONSHIP_LIST = ['owner of', 'partner of', 'investor in', 'competitor']
ORG_PRODUCT_RELATIONSHIP_LIST = ['developer of']
PRODUCT_PRODUCT_RELATIONSHIP_LIST = ['competitor']
