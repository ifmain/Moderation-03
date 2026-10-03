"""Four nested policies over 11 raw outputs plus a separate explicit profanity rule."""
import re
import unicodedata
from categories import CATEGORIES

LANGS='en de fr es it sv fi pl cs lv zh ja ko ru uk be kk'.split()
PROFILES=['light','medium','high','corporate']
# These are policy definitions, not measured model probability thresholds.
REFERENCE={
 'light':[None,.60,.80,.50,None,.60,None,None,.50,None,.60],
 'medium':[.55,.45,.60,.40,.85,.45,.85,.80,.40,.85,.45],
 'high':[.30,.30,.40,.25,.65,.30,.65,.50,.25,.60,.30],
 'corporate':[.15,.20,.20,.15,.45,.20,.45,.20,.15,.30,.20],
}
TERMS={
 'en':['fuck','fucking','shit','bullshit','motherfucker'],
 'de':['scheiße','scheisse','verdammt','arschloch'],
 'fr':['putain','merde','bordel'], 'es':['mierda','joder','coño'],
 'it':['cazzo','merda','vaffanculo'], 'sv':['jävla','jävlar','helvete'],
 'fi':['vittu','perkele','saatana'], 'pl':['kurwa','kurwy','pierdolić'],
 'cs':['kurva','kurvy','do prdele'], 'lv':['bļeģ','bļadj','pizģec'],
 'zh':['他妈的','操你妈','傻逼'], 'ja':['くそ','クソ','ちくしょう'],
 'ko':['씨발','시발','좆'], 'ru':['блядь','блять','хуй','хуя','пиздец','ебать','ебаный'],
 'uk':['блядь','блять','хуй','хуя','пиздець','їбаний'],
 'be':['блядзь','блядь','хуй','піздзец'], 'kk':['қотақ','блядь','пиздец'],
}
HOMO=str.maketrans({'а':'a','е':'e','ё':'e','о':'o','р':'p','с':'c','х':'x','у':'y','і':'i'})


def norm(text):
    return unicodedata.normalize('NFKC',text).casefold().translate(HOMO)


def pattern(word, lang):
    # Inserted separators and common digit substitutions are accepted inside known words.
    variants={'a':'a4','e':'e3','i':'i1','o':'o0','s':'s5','t':'t7','b':'b8','g':'g9','z':'z2'}
    letters=[]
    for ch in norm(word):
        if ch.isspace():continue
        letters.append('['+re.escape(variants.get(ch,ch))+']')
    core=r'[\s_.\-]*'.join(letters)
    return re.compile(core if lang in {'zh','ja','ko'} else r'(?<!\w)'+core+r'(?!\w)')

PATTERNS={lang:[pattern(word,lang) for word in terms] for lang,terms in TERMS.items()}


def profanity(text,language):
    normalized=norm(text)
    langs=[language] if language in PATTERNS else LANGS
    return any(p.search(normalized) for lang in langs for p in PATTERNS[lang])


def decisions(scores, profane, thresholds=None):
    thresholds=thresholds or {p:dict(zip(CATEGORIES,REFERENCE[p])) for p in PROFILES}
    result={}
    for profile in PROFILES:
        reasons=[c for c in CATEGORIES if thresholds[profile][c] is not None and scores[c]>=thresholds[profile][c]]
        if profane and profile in {'high','corporate'}:reasons.append('profanity_rule')
        result[profile]={'block':bool(reasons),'reasons':reasons}
    return result
