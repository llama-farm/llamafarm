import React from 'react';
import Translate from '@docusaurus/Translate';
import EditIcon from '../../components/icons/EditIcon';

export default function EditThisPage({editUrl}: {editUrl: string}): JSX.Element {
  return (
    <a
      href={editUrl}
      target="_blank"
      rel="noreferrer noopener"
      className="theme-edit-this-page">
      <EditIcon className="theme-edit-this-page-icon" />
      <Translate
        id="theme.common.editThisPage"
        description="The link label to edit the current page">
        Edit this page
      </Translate>
    </a>
  );
}

