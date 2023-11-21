from bionlp.processors import Entities, DiseaseProcessor, ChemicalProcessor, GeneProcessor
import spacy
from spacy.tokens import Doc
from spacy import util
import re
from bionlp.processors.utils import check_existant_model
from spacy.language import Language


#spacy.cli.download("en_core_web_sm")
def paragraphs(document):
    start = 0
    for token in document:
        if token.is_space and token.text.count("\n") > 1:
            yield document[start:token.i]
            start = token.i
    yield document[start:]


def process_by_paragraph(doc, entities):
    offset = 0
    for paragraph in paragraphs(doc):
        # print(len(str(paragraph)))
        disease_service.sentence_to_process(str(paragraph))
        disease_results = disease_service.predict()
        entities.append_new_entities(disease_results)
     

        entities.remove_non_entities()

        offset += len(str(paragraph))
        disease_service.set_offset(offset)
   
    disease_service.set_offset(0, restart=True)



@Language.factory("ner_custom")
def create_ner_model(nlp: Language, name: str):
    return NERComponent(nlp)


class NERComponent:
    def __init__(self, nlp: Language):
        self.entities = None

    def __call__(self, doc: Doc) -> Doc:
        self.entities = Entities(doc)
        process_by_paragraph(doc, self.entities)
        self.entities.postprocessing()
        return doc


@Language.component("postprocessing_covid")
def expand_covid_ents(doc):
    pattern_sars = r"((sarsr?|mers)(\s?\-?\s?(covs?))?(\s?\-?\s?2)?(\s?\binfe.{1,10}?\b)?)"
    pattern_covid = r"((covid)(\s?\-?\s?(19))?(\s?\binfe.{1,10}?\b)?)"
    pattern_coronavirus = r"((coronavir.{0,6}?\b)(\s?\bpneumo.{0,8}?\b)?(\s?\binfe.{1,10}?\b)?(\s?\bdiseas.{1,6}?\b)?(\s?\-?\s?(20)?(19))?)"
    pattern_variant_lineage = r"(\b[A-Z]{1}\.\d{1,4}(\.\d{1,4}){0,4}\b)"

    patterns_covid = [{"label": "DISEASE", "pattern": pattern_sars, "id": "covid"},
                      {"label": "DISEASE", "pattern": pattern_covid, "id": "covid"},
                      {"label": "DISEASE", "pattern": pattern_coronavirus, "id": "covid"},
                      {"label": "COVID LINEAGE", "pattern": pattern_variant_lineage, "id": "covid"}]

    new_ents = []
    doc_ents = list(doc.ents)
    for pattern in patterns_covid:
        for match in re.finditer(pattern['pattern'], doc.text, re.IGNORECASE):
            start, end = match.span()
            span = doc.char_span(start, end, label=pattern['label'], alignment_mode='expand')
            # This is a Span object or None if match doesn't map to valid token sequence
            if span is not None:
                #                 print((span.text, span.label_, span.start, span.end))
                new_ents.append(span)
    #                 print("Found match:", span.text)
    ents = doc_ents + new_ents
    filtered_spans = util.filter_spans(ents)
    doc.set_ents(filtered_spans)
    return doc


try:

    print('Loading DISEASE Service')
    if check_existant_model('Disease'):
        disease_service = DiseaseProcessor('./models/Disease')
    else:
        disease_service = DiseaseProcessor('alvaroalon2/biobert_diseases_ner')
    print('Disease service loaded')
  
    nlp = spacy.load("en_core_web_sm", exclude=["tok2vec", "lemmatizer"])

    nlp.add_pipe('ner_custom', before='ner')

    nlp.add_pipe('postprocessing_covid', before='ner')


    print('SYSTEM LOADED!!!')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

except Exception as e:
    print(repr(e))
    print('Error loading system components')
