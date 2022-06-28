from deardr.frontend.fever import FEVERPageLevelReader, FEVERPageLevelReaderSkipNEI
from deardr.frontend.hover import HOVERPageLevelReader
from deardr.frontend.kilt import KILTPageLevelReader
from deardr.frontend.yahoo import YahooAbstractReader
from deardr.frontend.yahoo_sentsplit import YahooAbstractSentSplitReader
from deardr.frontend.pretrain import PretrainPT, PretrainPTHLFiltered, PretrainHL, PretrainPTHL, PretrainHLFiltered

frontend_types = {
    "yahoo": YahooAbstractReader,
    "yahoosent": YahooAbstractSentSplitReader,
    "fever": FEVERPageLevelReaderSkipNEI,
    "fevertest": FEVERPageLevelReader,
    "hover": HOVERPageLevelReader,
    "kilt": KILTPageLevelReader,
    "pretrain_pt": PretrainPT,
    "pretrain_hl": PretrainHL,
    "pretrain_pthl": PretrainPTHL,
    "pretrain_hl_filter": PretrainHLFiltered,
    "pretrain_pthl_filter": PretrainPTHLFiltered,
}