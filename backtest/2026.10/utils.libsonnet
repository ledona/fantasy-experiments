{
  stacks_to_str(stacks)::
    std.join('', [std.toString(x) for x in stacks]),
  score_type_post(score_type)::
    if score_type != 'predicted' then ':' + score_type else '',
}
