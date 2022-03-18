def _yes_or_no(answer):
    answer = answer.lower()
    while answer not in {*_ringity_parameters['yes'], *_ringity_parameters['no']}:
        answer = input("Please provide a readable input, e.g. 'y' or 'n'! ")

    if answer in _ringity_parameters['yes']:
        return True
    elif answer in _ringity_parameters['no']:
        return False
    else:
        assert False, _assertion_statement

_assertion_statement =  "This should never happen, but apparently it does. " \
                        "Please contact mk.youssef@hotmail.com if you " \
                        "encounter this in the wild. Thanks!"
