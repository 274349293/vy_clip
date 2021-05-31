# encoding=utf-8
import re
from bs4 import BeautifulSoup


def remove_line_scape(string):
    return re.sub("\n", "", string)


def remove_html_tags(string):
    from bs4 import BeautifulSoup
    bs = BeautifulSoup(string, "html.parser")
    return bs.text


def merge_space(string):
    return re.sub(r"\s+", " ", string)


def remove_url(string):
    return re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", string)


def remove_control_chr(string):
    """"""
    import unicodedata

    string = "".join(ch for ch in string if ch in ['\r', '\n', ' '] or unicodedata.category(ch)[0] not in ["C", "Z"])
    return string


def remove_html_string(string):
    return re.sub(r"(?is)(?<=<a).*?(?=</a>)", "", string)


def split_document(document):
    chapters = document.split('\n')
    sentences = []
    for chapter in chapters:
        if "。" in chapter:
            for x in chapter.split('。'):
                sentences.append(x)
        else:
            sentences.append(chapter)
    return sentences


if __name__ == '__main__':
    import unicodedata

    print(unicodedata.category('\u3000'))
    print(unicodedata.category('\u4e2d'))
    print(unicodedata.category('\u200b'))
    print('\u200d'.isprintable())
    print('\u4e2d'.isprintable())
    print(('\u200d'.encode('utf-8')))
    print(ord('\u200b'))
    # exit()

    # insert_data_to_db(db_name='data-mining', collection_name='xinhua_tokens', data={"_id": 123, "tokens": ['1', '2']})
    _s = """
\u3000\u3000“昨天买的今天就收到了，一下子买了三件，金丝砗磲感觉很特别……”在京东上，一位买家在购买了砗磲制品后，不仅在评论区发图炫耀，还洋洋洒洒写下购买体验。\u3000\u3000砗磲是海洋中的一种贝类生物，由于人类的滥捕滥杀，现已濒临灭绝。\u3000\u30002018年，农业农村部发布第69号公告，再次明确砗磲科所有种均需按照国家重点保护动物级别进行国内管理，进出口环节需同时遵守国际公约有关规定。\u3000\u3000而早在1988年，库氏砗磲就已经被我国列为国家一级保护动物，严格禁止其制品的非法交易。\u3000\u3000在拼多多上，一些卖家打着“泡酒中药材”的名义兜售野生动物制成品。在一家店铺内，卖家展示了包括“蛤蚧粉”“蛤蚧干”和“整条白花蛇（金钱蛇）”在内的多种野生动物制品。蛤蚧是大壁虎的俗称，是国家二级保护动物。一位售卖“蛤蚧粉”的卖家表示，其售卖的“蛤蚧粉”原料来源于野生大壁虎。\u3000\u3000在淘宝搜索“砗磲”时，平台会自动给用户跳转至“绿网计划”的公益广告界面，但当点击搜索栏下方系统给出的搜索结果时，砗磲制品又能够“琳琅满目”地呈现在眼前，平台的AI好像完全“知晓”买家的购买意图。\u3000\u3000还有一些商家，在电商平台上以“防鸟网”的名义兜售捕鸟网。据野生动物保护志愿者张晓磊介绍，对野生鸟类而言，捕鸟网较难被察觉，一旦撞上捕鸟网，如果不借助外力，鸟类很难自行脱困，大部分鸟类都会惨死在捕鸟网上。\u3000\u3000志愿者反映，淘宝等电商平台对于他们的举报大多予以驳回处理，即使上传了与不法商家的聊天截图，并且使用多个账号反复举报，平台也不予认可。\u3000\u3000此外，记者也尝试对上述三家平台中国家明文禁止交易的野生动物制品和非法猎捕工具进行举报。\u3000\u3000结果显示，淘宝App不到4小时即给出了“举报不成立”的结论，且官方客服没有通过任何方式联系举报人进行线索核实；京东客服在记者电话举报45小时后告知记者先暂时下架相关商品，并将开展线索核实工作，后续将视情决定是否禁止卖家重新上架相关商品；拼多多方面在接到线索举报后多日内未对违规商品做出处理。\u3000\u3000截至记者发稿，平台方面也没有主动联系记者进一步核实或申明处理意见。\u3000\u3000目前，记者已将调查所获取的问题线索转交相关平台，并将涉及违法犯罪的相关证据移交有关部门。记者：颜之宏编辑：王朝新华社音视频部制作
    """
    import re

    test = '<p>阐述均以图片形[赞同]式呈现，方面各位[赞同]保存给家里被</p>骗老人看[IMGatHere][IMGatHere]<b>[IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere]关于清真认证到底是不是那么牛逼，各位，神奇的传送门：清真认证_百度百科身在一个穆斯林泛滥地区的答主我告诫各位！！！孕妇和宝宝不能用的不能吃的，就算是清真认证了也是不能用不能吃的！[IMGatHere][IMGatHere][IMGatHere][IMGatHere][IMGatHere'

    # _s = remove_control_chr(_s)
    # print([_s])
    # _s = _s.split('\n')
    # # print(_s)
    # _tokenizer = HanLPTokenizer()
    # for _sent in _s:
    #     if len(_sent) == 0:
    #         continue
    #     print(_sent)
    #     _tok = _tokenizer.tokenize(_sent)
    #     print(_tok)

    # remove_stop_words([], './stopwords/')

"""
[Cc] Other, Control

[Cf] Other, Format

[Cn] Other, Not Assigned (no characters in the file have this property)

[Co] Other, Private Use

[Cs] Other, Surrogate

[LC] Letter, Cased

[Ll] Letter, Lowercase

[Lm] Letter, Modifier

[Lo] Letter, Other

[Lt] Letter, Titlecase

[Lu] Letter, Uppercase

[Mc] Mark, Spacing Combining

[Me] Mark, Enclosing

[Mn] Mark, Nonspacing

[Nd] Number, Decimal Digit

[Nl] Number, Letter

[No] Number, Other

[Pc] Punctuation, Connector

[Pd] Punctuation, Dash

[Pe] Punctuation, Close

[Pf] Punctuation, Final quote (may behave like Ps or Pe depending on usage)

[Pi] Punctuation, Initial quote (may behave like Ps or Pe depending on usage)

[Po] Punctuation, Other

[Ps] Punctuation, Open

[Sc] Symbol, Currency

[Sk] Symbol, Modifier

[Sm] Symbol, Math

[So] Symbol, Other

[Zl] Separator, Line

[Zp] Separator, Paragraph

[Zs] Separator, Space
"""
