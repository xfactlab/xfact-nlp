from deardr.frontend.fever_reader import FEVERTestReader
from deardr.frontend.yahoo import YahooAbstractReader
from deardr.frontend.yahoo_sentsplit import YahooAbstractSentSplitReader
from deardr.frontend.pretrain import PretrainPT, PretrainPTHLFiltered, PretrainHL, PretrainPTHL, PretrainHLFiltered

frontend_types = {
    "yahoo": YahooAbstractReader,
    "yahoosent": YahooAbstractSentSplitReader,
    "fevertest": FEVERTestReader,
    "pretrain_pt": PretrainPT,
    "pretrain_hl": PretrainHL,
    "pretrain_pthl": PretrainPTHL,
    "pretrain_hl_filter": PretrainHLFiltered,
    "pretrain_pthl_filter": PretrainPTHLFiltered,
}