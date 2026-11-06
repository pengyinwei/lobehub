import { createStaticStyles, Flexbox, Text } from '@lobehub/ui';
import { useTranslation } from 'react-i18next';
import { useParams } from 'react-router';

import IntegrationsSettings from '@/features/Integrations';
import SettingHeader from '@/features/Settings/features/SettingHeader';

const styles = createStaticStyles(({ css }) => ({
  header: css`
    width: 100%;
    max-width: 760px;
    margin-inline: auto;
  `,
}));

interface PageProps {
  showSettingHeader?: boolean;
}

const Page = ({ showSettingHeader = true }: PageProps) => {
  const { t } = useTranslation('integration');
  const params = useParams<{ sub?: string }>();
  // The directory carries the page intro; an integration page draws its own hero.
  // Under the shared settings header only the description stays in the body.
  const showIntro = !params.sub;

  if (showSettingHeader)
    return (
      <>
        {showIntro && (
          <div className={styles.header}>
            <SettingHeader description={t('overview.description')} title={t('overview.title')} />
          </div>
        )}
        <IntegrationsSettings />
      </>
    );

  // Intro line and directory read as one block, like the messenger list.
  return (
    <Flexbox gap={20}>
      {showIntro && (
        <div className={styles.header}>
          <Text type={'secondary'}>{t('overview.description')}</Text>
        </div>
      )}
      <IntegrationsSettings />
    </Flexbox>
  );
};

export default Page;
