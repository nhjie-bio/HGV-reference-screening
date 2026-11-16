#!/usr/bin/env python3
from common import confirmation_status_class
assert confirmation_status_class('OK') == 'REPORTED'
assert confirmation_status_class('NO_HIT_DEFAULT_AF') == 'NO_HIT'
assert confirmation_status_class('NO_HIT_LOW_ANI') == 'NO_HIT'
assert confirmation_status_class('EXEC_FAIL:x') == 'HARD_FAIL'
assert confirmation_status_class('PARSE_FAIL:x') == 'HARD_FAIL'
assert confirmation_status_class('TIMEOUT') == 'HARD_FAIL'
assert confirmation_status_class('MISSING') == 'HARD_FAIL'
print('CONFIRMATION_STATUS_UNIT_TEST=PASS')
