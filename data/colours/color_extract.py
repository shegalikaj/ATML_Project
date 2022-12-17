

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fields = ['color', 'hexadecimal','R', 'G', 'B']

df = pd.read_csv('colors.csv', usecols = fields, low_memory = True)
df['RGB'] = list(zip(df.R, df.G, df.B))

colors = ["almond", "amaranth", "amber", "amethyst", "antique brass", "antique fuchsia", "antique ruby", "antique white", "apricot", "aqua", "aquamarine", "arsenic", "asparagus", "auburn", "aureolin", "aurometalsaurus", "avocado", "azure", "baby blue", "baby blue eyes", "baby pink", "bazaar", "beaver", "beige", "bisque", "bistre", "bittersweet", "black", "blond", "blue", "blush", "bole", "bone", "boysenberry", "brass", "bright cerulean", "bright green", "bright lavender", "bright maroon", "bright pink", "bright turquoise", "bright ube", "bronze", "bubbles", "buff", "burgundy", "burlywood", "burnt orange", "burnt sienna", "burnt umber", "byzantine", "byzantium", "cadet", "camel", "capri", "cardinal", "carmine", "carnelian", "ceil", "celadon", "cerise", "cerulean", "chamoisee", "champagne", "charcoal", "cherry", "chestnut", "cinereous", "cinnabar", "cinnamon", "citrine", "cobalt", "coffee", "copper", "copper crayola", "copper penny", "copper red", "copper rose", "coquelicot", "coral", "cordovan", "corn", "cornsilk", "cream", "crimson","daffodil", "dandelion", "dark blue", "dark brown", "dark byzantium", "dark cerulean", "dark chestnut", "dark coral", "dark cyan", "dark electric blue", "dark goldenrod", "dark gray", "dark green", "dark imperial blue", "dark jungle green", "dark khaki", "dark lava", "dark lavender", "dark magenta", "dark midnight blue", "dark olive green", "dark orange", "dark orchid", "dark pastel blue", "dark pastel green", "dark pastel purple", "dark pastel red", "dark pink", "dark powder blue", "dark raspberry", "dark red", "dark salmon", "dark scarlet", "dark sea green", "dark sienna", "dark slate blue", "dark slate gray", "dark spring green", "dark tan", "dark tangerine", "dark taupe", "dark terra cotta", "dark turquoise", "dark violet", "dark yellow", "deep carmine", "deep carmine pink", "deep carrot orange", "deep cerise", "deep champagne", "deep chestnut", "deep coffee", "deep fuchsia", "deep jungle green", "deep lilac", "deep magenta", "deep peach", "deep pink", "deep ruby", "deep saffron", "deep sky blue", "deep tuscan red", "denim", "desert", "drab", "ebony", "ecru", "eggplant", "eggshell", "electric blue", "electric crimson", "electric cyan", "electric green", "electric indigo", "electric lavender", "electric lime", "electric purple", "electric ultramarine", "electric violet", "electric yellow", "emerald", "fallow", "fandango", "fawn", "feldgrau", "firebrick", "flame", "flavescent", "flax", "fluorescent orange", "fluorescent pink", "fluorescent yellow", "folly", "french beige", "french blue", "french lilac", "french lime", "french raspberry", "french rose", "fuchsia", "fulvous", "gainsboro", "gamboge", "ginger", "glaucous", "glitter", "goldenrod", "gray", "grullo", "harlequin", "heliotrope", "honeydew", "iceberg", "icterine", "inchworm", "indigo", "iris", "isabelline", "ivory", "jade", "jasmine", "jasper", "jet", "jonquil", "lava", "lemon", "licorice", "light apricot", "light blue", "light brown", "light carmine pink", "light coral", "light cornflower blue", "light crimson", "light cyan", "light fuchsia pink", "light goldenrod yellow", "light gray", "light green", "light khaki", "light pastel purple", "light pink", "light red ochre", "light salmon", "light salmon pink", "light sea green", "light sky blue", "light slate gray", "light taupe", "light thulian pink", "light yellow", "lilac", "limerick", "linen", "lion", "liver", "lust", "magnolia", "mahogany", "maize", "malachite", "manatee", "mantis", "mauve", "mauvelous", "medium aquamarine", "medium blue", "medium carmine", "medium champagne", "medium electric blue", "medium jungle green", "medium lavender magenta", "medium orchid", "medium persian blue", "medium purple", "medium red violet", "medium ruby", "medium sea green", "medium slate blue", "medium spring bud", "medium spring green", "medium taupe", "medium turquoise", "medium tuscan red", "medium vermilion", "medium violet red", "melon", "mint", "moccasin", "mulberry", "mustard", "myrtle", "neon carrot", "neon fuchsia", "neon green", "ochre", "old gold", "old lace", "old lavender", "old mauve", "old rose", "olive", "olivine", "onyx", "orange", "orchid", "pale aqua", "pale blue", "pale brown", "pale carmine", "pale cerulean", "pale chestnut", "pale copper", "pale cornflower blue", "pale gold", "pale goldenrod", "pale green", "pale lavender", "pale magenta", "pale pink", "pale plum", "pale red violet", "pale silver", "pale spring bud", "pale taupe", "pale violet red", "pastel blue", "pastel brown", "pastel gray", "pastel green", "pastel magenta", "pastel orange", "pastel pink", "pastel purple", "pastel red", "pastel violet", "pastel yellow", "patriarch", "peach", "pear", "pearl", "peridot", "periwinkle", "persian blue", "persian green", "persian indigo", "persian orange", "persian pink", "persian plum", "persian red", "persian rose", "persimmon", "peru", "phlox", "pink", "pistachio", "platinum", "prune", "puce", "pumpkin", "quartz", "rackley", "rajah", "raspberry", "razzmatazz", "red", "redwood", "regalia", "rich black", "rich brilliant lavender", "rich carmine", "rich electric blue", "rich lavender", "rich lilac", "rich maroon", "rose", "rose bonbon", "rose ebony", "rose gold", "rose madder", "rose pink", "rose quartz", "rose taupe", "rose vale", "rosewood", "royal azure", "royal blue traditional", "royal blue web", "royal fuchsia", "royal purple", "royal yellow", "ruby", "ruddy", "rufous", "russet", "rust", "saffron", "salmon", "sand", "sandstorm", "sangria", "sapphire", "scarlet", "seashell", "sepia", "shadow", "sienna", "silver", "sinopia", "skobeloff", "snow", "stizza", "stormcloud", "straw", "sunglow", "sunset", "tan", "tangelo", "tangerine", "taupe", "teal", "telemagenta", "thistle", "timberwolf", "tomato", "toolbox", "topaz", "tumbleweed", "turquoise", "ube", "ultramarine", "umber", "urobilin", "vanilla", "verdigris", "veronica", "violet", "viridian", "vivid auburn", "vivid burgundy", "vivid cerise", "vivid tangerine", "vivid violet", "waterspout", "wenge", "wheat", "white", "wine", "wisteria", "xanadu","zaffre", "green"]

all_colors = list(df.color)
r = list(df.R)
g = list(df.G)
b = list(df.B)
hexadecimal = list(df.hexadecimal)


for key, value in enumerate(all_colors): 
  value.lower()
  if value.lower not in colors:
    all_colors.remove(value)
    r.pop(key)
    g.pop(key)
    b.pop(key)
    hexadecimal.pop(key)


import pandas as pd

data = {'color': all_colors,
         'hexadecimal': hexadecimal,
         'R': r,
         'G': g,
         'B': b
        }

df = pd.DataFrame.from_dict(data)

df.to_csv("extracted_colors.csv")