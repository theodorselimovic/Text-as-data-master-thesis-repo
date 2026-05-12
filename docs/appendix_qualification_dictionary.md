# Appendix: Risk Qualification Dictionary

This appendix documents the qualification dictionary used for context analysis of Swedish Risk and Vulnerability Analyses (RSA). The dictionary captures how risks are qualified in terms of probability, consequence, and overall risk level. The script first looks through the text for sentences containing target words related to probability, consequence, and risk. It then counts qualifiers in the sentences. For intensity, qualifiers are grouped into 5 categories: very low, low, medium, high, very high. For direction, there are 3 categories: increasing, decreasing, and stable. Multi-word qualifiers (e.g., "mycket hög") take precedence over single-word matches.

## 1. Target Words

| Concept | Swedish Variants |
|---------|------------------|
| sannolikhet (probability) | sannolikhet, sannolikheten, sannolikhets, trolig, troligt, troliga, sannolik, sannolikt, sannolika, osannolik, osannolikt, osannolika |
| konsekvens (consequence) | konsekvens, konsekvensen, konsekvenser |
| risk | risk, risken, risker, riskens |

## 2. Intensity Qualifiers

### Very Low

| Concept | Swedish Terms | English |
|---------|---------------|---------|
| sannolikhet | mycket låg, mycket liten, sällsynt | very low, very small, rare |
| konsekvens | mycket begränsade, mycket liten, försumbara, obetydlig, obetydliga, marginell, marginella, minimal, minimala | very limited, negligible, insignificant, marginal, minimal |
| risk | mycket låg, mycket liten, försumbar, obetydlig, minimal | very low, very small, negligible, insignificant, minimal |

### Low

| Concept | Swedish Terms | English |
|---------|---------------|---------|
| sannolikhet | låg, liten, små | low, small |
| konsekvens | begränsade, lindriga, liten, små, ringa, måttlig, måttliga | limited, mild, small, slight, moderate |
| risk | låg, liten, små, begränsad, måttlig | low, small, limited, moderate |

### Medium

| Concept | Swedish Terms | English |
|---------|---------------|---------|
| sannolikhet | medelhög, mellan, möjlig | medium-high, intermediate, possible |
| konsekvens | kännbara, måttliga, direkta, märkbar, märkbara, påtaglig, påtagliga | noticeable, moderate, direct, tangible |
| risk | medelhög, mellan, påtaglig, märkbar | medium-high, intermediate, tangible, noticeable |

### High

| Concept | Swedish Terms | English |
|---------|---------------|---------|
| sannolikhet | hög, stor | high, large |
| konsekvens | allvarlig, allvarliga, betydande, stor, stora, omfattande, svåra, kraftig, kraftiga, avsevärd, avsevärda, väsentlig, väsentliga | serious, significant, large, extensive, severe, considerable, substantial |
| risk | hög, stor, stora, omfattande, avsevärd, betydande, väsentlig | high, large, extensive, considerable, significant, substantial |

### Very High

| Concept | Swedish Terms | English |
|---------|---------------|---------|
| sannolikhet | mycket hög, mycket stora, stora | very high, very large |
| konsekvens | mycket allvarlig, mycket allvarliga, mycket stora, mycket omfattande, katastrofal, katastrofala, extrem, extrema, förödande, ödesdigra | very serious, very large, very extensive, catastrophic, extreme, devastating, fatal |
| risk | mycket hög, mycket stora, mycket omfattande, extrem, kritisk, akut, överhängande | very high, very large, very extensive, extreme, critical, acute, imminent |

## 3. Direction Qualifiers

### Increasing

| Swedish Terms | English |
|---------------|---------|
| öka, ökar, ökad, ökande | increase, increasing, increased |
| stiger, stigande | rise, rising |
| tilltar, tilltagande | intensify, intensifying |
| förvärras, förvärrad | worsen, worsened |
| eskalerar, eskalerande | escalate, escalating |
| förhöjd, förhöjda | elevated |
| växer, växande | grow, growing |
| förstärkt, förstärkta | strengthened, intensified |
| skärpt, skärpta | heightened, tightened |
| accelererar | accelerating |

### Decreasing

| Swedish Terms | English |
|---------------|---------|
| minska, minskar, minskad, minskande | decrease, decreasing, decreased |
| sjunker, sjunkande | sink, sinking |
| avtar, avtagande | diminish, diminishing |
| reducerad, reduceras | reduced |
| lindras, lindrad | alleviate, alleviated |
| förbättras, förbättrad | improve, improved |
| sänks, sänkt | lower, lowered |
| mildras, mildrad | mitigate, mitigated |
| avklingande | subsiding |
| dämpas, dämpad | dampened |

### Stable

| Swedish Terms | English |
|---------------|---------|
| stabil, stabilt, stabila | stable |
| oförändrad, oförändrat, oförändrade | unchanged |
| konstant, konstanta | constant |
| bibehållen, bibehållet, bibehållna | maintained, retained |
| kvarstår, kvarstående | remain, remaining |
| bestående, beständig | persistent, lasting |
| varaktig, varaktigt, varaktiga | lasting, enduring |
| ihållande | persistent, sustained |
| fortsatt, fortsatta | continued |
| orubblig, orubbligt | unwavering, steadfast |
| jämn, jämnt, jämna | steady, even |
| opåverkad, opåverkat, opåverkade | unaffected |
| likvärdig, likvärdigt, likvärdiga | equivalent, same level |
| samma | same |

## 4. Summary Statistics

- **Target concepts:** 3
- **Qualification levels:** 8 (5 magnitude + 3 trend)
- **Total unique terms:** ~140
- **Increasing terms:** 19
- **Decreasing terms:** 19
- **Stable terms:** 27
